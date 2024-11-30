use std::{
    cmp::Ordering,
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex, Weak},
};

use anyhow::{bail, Result};
use half::f16;
use itertools::Itertools;
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
};
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    num::{CoHom, Float},
    runtime::{
        infer::{InferInput, InferInputBatch, InferOption, InferOutput, InferOutputBatch},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant},
        v6, TokioRuntime,
    },
    tensor::{kind::ReadWrite, ops::TensorOp, TensorCpu, TensorGpu, TensorShape},
    tokenizer::Tokenizer,
    wgpu,
};

const MAX_INPUT_TOKENS: usize = 4096;
const TOKEN_CHUNK_SIZE: usize = 128;
const PREDICT_TOP_K: usize = 16;

#[derive(Debug, Clone)]
struct Runtime {
    tokenizer: Arc<Tokenizer>,
    runtime: TokioRuntime<InferInput, InferOutput>,
    model: v6::Model,
    data: Vec<HookDataGpu>,
}

#[derive(Debug, Clone)]
struct HookDataGpu {
    k: TensorGpu<f32, ReadWrite>,
    v: TensorGpu<f32, ReadWrite>,
    r: TensorGpu<f32, ReadWrite>,
    w: TensorGpu<f32, ReadWrite>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct HookDataCpu {
    k: Vec<TensorCpu<f32>>,
    v: Vec<TensorCpu<f32>>,
    r: Vec<TensorCpu<f32>>,
    w: Vec<TensorCpu<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Pack {
    info: ModelInfo,
    tokens: Vec<u16>,
    decoded: Vec<String>,
    data: Vec<HookDataCpu>,
    predicts: Vec<[(u16, String, f32); PREDICT_TOP_K]>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct HeadKey {
    layer: usize,
    head: usize,
    source: usize,
    token: usize,
}

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = web_rwkv::wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .auto_limits(info)
        .build()
        .await?;
    Ok(context)
}

async fn load_tokenizer() -> Result<Tokenizer> {
    let file = File::open("assets/rwkv_vocab_v20230424.json").await?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents).await?;
    Ok(Tokenizer::new(&contents)?)
}

async fn load_runtime(
    context: &Context,
    info: &ModelInfo,
    data: &[u8],
    quant: HashMap<usize, Quant>,
) -> Result<Runtime> {
    if !matches!(info.version, ModelVersion::V6) {
        bail!("only supports v6 models")
    }

    let tokenizer = Arc::new(load_tokenizer().await?);

    let model = SafeTensors::deserialize(data)?;
    let model = ModelBuilder::new(context, model)
        .quant(quant)
        .build_v6()
        .await?;

    let data = (0..model.info.num_layer)
        .map(|_| {
            let context = &model.context;
            let num_emb = model.info.num_emb;
            let num_token = TOKEN_CHUNK_SIZE;
            let shape = [num_emb, num_token, 1, 1];
            HookDataGpu {
                k: context.tensor_init(shape),
                v: context.tensor_init(shape),
                r: context.tensor_init(shape),
                w: context.tensor_init(shape),
            }
        })
        .collect_vec();

    let mut hooks: v6::HookMap<f16> = v6::HookMap::new();
    for (n, data) in data.iter().cloned().enumerate() {
        hooks.insert(
            v6::Hook::PostAtt(n),
            Box::new(move |frame: v6::Frame<f16>| {
                let num_token = frame.buffer.att_x.shape()[1];

                let ops = vec![
                    TensorOp::blit(
                        frame.buffer.att_k.view(.., .., .., ..)?,
                        data.k.view(.., ..num_token, .., ..)?,
                    )?,
                    TensorOp::blit(
                        frame.buffer.att_v.view(.., .., .., ..)?,
                        data.v.view(.., ..num_token, .., ..)?,
                    )?,
                    TensorOp::blit(
                        frame.buffer.att_r.view(.., .., .., ..)?,
                        data.r.view(.., ..num_token, .., ..)?,
                    )?,
                    TensorOp::blit(
                        frame.buffer.time_decay.view(.., .., .., ..)?,
                        data.w.view(.., ..num_token, .., ..)?,
                    )?,
                ];

                Ok(TensorOp::List(ops))
            }),
        );
    }

    let bundle = v6::Bundle::new_with_hooks(model.clone(), 1, hooks);
    let runtime = TokioRuntime::new(bundle).await;

    Ok(Runtime {
        tokenizer,
        runtime,
        model,
        data,
    })
}

type BoxUi = Box<dyn Fn(&egui::Context, &mut eframe::Frame) + Send + Sync>;

#[derive(Debug, Clone)]
struct UiHandle(Arc<()>);

#[derive(Debug, Clone)]
struct Ui(flume::Sender<(BoxUi, Weak<()>)>);

impl Ui {
    fn create<F>(&self, ui: F) -> UiHandle
    where
        F: Fn(&egui::Context, &mut eframe::Frame) + Send + Sync + 'static,
    {
        let handle = UiHandle(Arc::new(()));
        let weak = Arc::downgrade(&handle.0);
        let _ = self.0.send((Box::new(ui), weak));
        handle
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
struct UiId;

struct App {
    rx: flume::Receiver<(BoxUi, Weak<()>)>,
    ui: Vec<(BoxUi, Weak<()>)>,
}

impl App {
    fn new() -> (Self, Ui) {
        let (tx, rx) = flume::unbounded();
        let ui = vec![];
        let app = Self { rx, ui };
        (app, Ui(tx))
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        while let Ok(ui) = self.rx.try_recv() {
            self.ui.push(ui);
        }

        for ui in self.ui.iter() {
            (ui.0)(ctx, frame);
        }

        let mut retain = vec![];
        for ui in self.ui.drain(..) {
            if ui.1.upgrade().is_some() {
                retain.push(ui);
            }
        }
        self.ui = retain;
    }
}

async fn run(ui: Ui) -> Result<()> {
    let runtime = 'load: loop {
        let path = {
            let (tx, rx) = flume::unbounded();
            let _ui_load = ui.create(move |ctx, _| {
                egui::Window::new("Model").title_bar(false).show(ctx, |ui| {
                    ui.label("Choose a model file.");
                    ui.separator();
                    if ui.button("Open...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("SafeTensors", &["st"])
                            .pick_file()
                        {
                            let _ = tx.send(path);
                        }
                    }
                });
            });
            rx.recv_async().await?
        };

        let ui_load = {
            let path = path.clone();
            ui.create(move |ctx, _| {
                egui::Window::new("Load").title_bar(false).show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(format!("Loading model from {}...", path.to_string_lossy()));
                        ui.spinner();
                    });
                });
            })
        };

        let file = File::open(path).await?;
        let data = unsafe { Mmap::map(&file)? };

        let model = SafeTensors::deserialize(&data)?;
        let info = Loader::info(&model)?;
        log::info!("{:#?}", info);

        let quant = {
            let (tx, rx) = flume::unbounded();
            let quant = Mutex::new(0usize);
            let quant_nf4 = Mutex::new(0usize);
            let _load_ui = ui.create(move |ctx, _| {
                use egui::{Slider, Window};

                let mut quant = quant.lock().unwrap();
                let mut quant_nf4 = quant_nf4.lock().unwrap();

                Window::new("Configure").show(ctx, |ui| {
                    ui.add(Slider::new(&mut *quant, 0..=info.num_layer).text("Int8 Layers"));
                    ui.add(Slider::new(&mut *quant_nf4, 0..=info.num_layer).text("NF4 Layers"));

                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Load").clicked() {
                            let quant = (0..*quant)
                                .map(|layer| (layer, Quant::Int8))
                                .chain((0..*quant_nf4).map(|layer| (layer, Quant::NF4)))
                                .collect();
                            let _ = tx.send(Some(quant));
                        }
                        if ui.button("Back").clicked() {
                            let _ = tx.send(None);
                        }
                    })
                });
            });

            match rx.recv_async().await? {
                Some(quant) => quant,
                _ => continue 'load,
            }
        };

        let context = create_context(&info).await?;
        log::info!("{:#?}", context.adapter.get_info());

        match load_runtime(&context, &info, &data, quant).await {
            Ok(runtime) => break 'load runtime,
            Err(err) => {
                drop(ui_load);
                let (tx, rx) = flume::unbounded();
                let _ui_load = ui.create(move |ctx, _| {
                    egui::Window::new("Load").title_bar(false).show(ctx, |ui| {
                        ui.label(format!("Error: {}", err));
                        if ui.button("Ok").clicked() {
                            let _ = tx.send(());
                        }
                    });
                });
                rx.recv_async().await?;
            }
        }
    };

    let _ui_info = info_ui(ui.clone(), runtime.model.info.clone()).await;

    'input: loop {
        let (tx, rx) = flume::unbounded();
        let (input_tx, input_rx) = tokio::sync::watch::channel(true);
        let text = Mutex::new(String::new());

        let ui_input = ui.create(move |ctx, _| {
            use egui::{ScrollArea, Window};

            Window::new("Input").max_height(400.0).show(ctx, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    if *input_rx.borrow() {
                        let mut text = text.lock().unwrap();
                        ui.text_edit_multiline(&mut *text);
                    } else {
                        let text = text.lock().unwrap();
                        ui.label(&*text);
                    }
                });
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.button("Submit").clicked() {
                        let text = text.lock().unwrap();
                        let _ = tx.send(text.clone());
                    }
                    if !*input_rx.borrow() {
                        ui.spinner();
                    }
                });
            });
        });

        let input = rx.recv_async().await?;
        let Ok(mut tokens) = runtime.tokenizer.encode(&input.into_bytes()) else {
            continue 'input;
        };
        tokens.truncate(MAX_INPUT_TOKENS);

        if tokens.is_empty() {
            continue 'input;
        }

        // invalidate the input ui
        let _ = input_tx.send(false);

        // list of tokens in the input
        let decoded = tokens
            .iter()
            .map(|&token| runtime.tokenizer.decode(&[token]).unwrap_or_default())
            .map(|x| String::from_utf8_lossy(&x).to_string())
            .collect_vec();

        let mut inference = {
            let tokens = tokens.clone();
            let option = InferOption::Full;
            Some(InferInput::new(
                vec![InferInputBatch { tokens, option }],
                TOKEN_CHUNK_SIZE,
            ))
        };

        let mut predicts = vec![];
        let mut data = vec![HookDataCpu::default(); runtime.model.info.num_layer];
        'prefill: loop {
            let input = inference.take().unwrap();
            let pre_num_token = input.batches[0].tokens.len();

            let (input, InferOutput(batches)) = runtime.runtime.infer(input).await?;
            let post_num_token = input.batches[0].tokens.len();
            inference = Some(input);

            fn split<F: Float>(x: TensorCpu<F>) -> Vec<TensorCpu<f32>> {
                x.split(1)
                    .expect("split batch")
                    .into_iter()
                    .map(|x| x.map(|&x| CoHom::co_hom(x)))
                    .collect()
            }

            let InferOutputBatch(batch) = batches.into_iter().next().unwrap();
            let num_token = pre_num_token - post_num_token;

            let batch = {
                let context = &runtime.model.context;
                web_rwkv::runtime::softmax::softmax_one(context, batch).await?
            };

            for x in split(batch).into_iter() {
                let x = x.to_vec();
                let x = x
                    .into_iter()
                    .enumerate()
                    .sorted_by(|x, y| x.1.total_cmp(&y.1).reverse())
                    .take(PREDICT_TOP_K)
                    .map(|(token, prob)| {
                        let token = token as u16;
                        let decoded = runtime.tokenizer.decode(&[token]).expect("decode token");
                        let decoded = String::from_utf8_lossy(&decoded).to_string();
                        (token, decoded, prob)
                    })
                    .collect_vec();
                predicts.push(x.try_into().expect("vec to array"));
            }

            for (cpu, gpu) in data.iter_mut().zip_eq(runtime.data.iter()) {
                let mut k = split(gpu.k.back().await);
                let mut v = split(gpu.v.back().await);
                let mut r = split(gpu.r.back().await);
                let mut w = split(gpu.w.back().await);
                k.truncate(num_token);
                v.truncate(num_token);
                r.truncate(num_token);
                w.truncate(num_token);
                cpu.k.append(&mut k);
                cpu.v.append(&mut v);
                cpu.r.append(&mut r);
                cpu.w.append(&mut w);
            }

            if inference
                .as_ref()
                .map(|input| input.batches[0].tokens.len())
                == Some(0)
            {
                break 'prefill;
            }
        }
        drop(ui_input);

        let info = runtime.model.info.clone();
        let pack = Pack {
            info,
            tokens,
            decoded,
            data,
            predicts,
        };

        'save: loop {
            #[derive(Debug, Clone, Serialize, Deserialize)]
            enum SaveOption {
                Save(PathBuf),
                Continue,
                Back,
            }

            let (tx, rx) = flume::unbounded();
            let _ui_save = ui.create(move |ctx, _| {
                egui::Window::new("Save").title_bar(false).show(ctx, |ui| {
                    ui.label("Save the trace?");
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Save...").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Trace", &["tr"])
                                .save_file()
                            {
                                let _ = tx.send(SaveOption::Save(path));
                            }
                        }
                        if ui.button("Continue").clicked() {
                            let _ = tx.send(SaveOption::Continue);
                        }
                        if ui.button("Back").clicked() {
                            let _ = tx.send(SaveOption::Back);
                        }
                    });
                });
            });

            let path = match rx.recv_async().await? {
                SaveOption::Save(path) => path,
                SaveOption::Continue => break 'save,
                SaveOption::Back => continue 'input,
            };

            let pack = pack.clone();
            let (tx, rx) = flume::unbounded();
            tokio::task::spawn_blocking(move || -> Result<()> {
                use std::{fs::File, io::Write};

                struct FileWriter(File);
                impl cbor4ii::core::enc::Write for FileWriter {
                    type Error = std::io::Error;

                    fn push(&mut self, input: &[u8]) -> Result<(), Self::Error> {
                        self.0.write_all(input)
                    }
                }

                let file = FileWriter(File::create(path)?);
                let mut serializer = cbor4ii::serde::Serializer::new(file);

                pack.serialize(&mut serializer)?;
                let _ = tx.send(());
                Ok(())
            });

            let _save_ui = ui.create(|ctx, _| {
                egui::Window::new("Save").title_bar(false).show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Saving trace file...");
                        ui.spinner();
                    });
                });
            });

            rx.recv_async().await?;
        }

        inspect(ui.clone(), pack).await?;
    }
}

async fn trace(ui: Ui) -> Result<()> {
    'trace: loop {
        let path = {
            let (tx, rx) = flume::unbounded();
            let _ui_load = ui.create(move |ctx, _| {
                egui::Window::new("Trace").title_bar(false).show(ctx, |ui| {
                    ui.label("Choose a trace file.");
                    ui.separator();
                    if ui.button("Open...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Trace", &["tr"])
                            .pick_file()
                        {
                            let _ = tx.send(path);
                        }
                    }
                });
            });
            rx.recv_async().await?
        };

        let pack = {
            let file = File::open(path).await?;
            let data = unsafe { Mmap::map(&file)? };

            match cbor4ii::serde::from_slice::<Pack>(&data) {
                Ok(pack) => pack,
                Err(err) => {
                    let (tx, rx) = flume::unbounded();
                    let _ui_load = ui.create(move |ctx, _| {
                        egui::Window::new("Trace").title_bar(false).show(ctx, |ui| {
                            ui.label(format!("Error: {}", err));
                            if ui.button("Ok").clicked() {
                                let _ = tx.send(());
                            }
                        });
                    });
                    rx.recv_async().await?;
                    continue 'trace;
                }
            }
        };

        let _ui_info = info_ui(ui.clone(), pack.info.clone()).await;

        inspect(ui.clone(), pack).await?;
    }
}

async fn info_ui(ui: Ui, info: ModelInfo) -> UiHandle {
    let ui_id = uid::Id::<UiId>::new();

    ui.create(move |ctx, _| {
        use egui::{Grid, Id, Window};

        Window::new("Info").id(Id::new(ui_id)).show(ctx, |ui| {
            Grid::new("grid")
                .num_columns(2)
                .spacing([40.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Version");
                    ui.label(format!("{:?}", info.version));
                    ui.end_row();

                    ui.label("Layer Count");
                    ui.label(format!("{}", info.num_layer));
                    ui.end_row();

                    ui.label("Vocab Size");
                    ui.label(format!("{}", info.num_vocab));
                    ui.end_row();

                    ui.label("Embed Size");
                    ui.label(format!("{}", info.num_emb));
                    ui.end_row();

                    ui.label("FFN Size");
                    ui.label(format!("{}", info.num_hidden));
                    ui.end_row();

                    ui.label("Head Count");
                    ui.label(format!("{}", info.num_head));
                    ui.end_row();

                    ui.label("Head Size");
                    ui.label(format!("{}", info.num_emb / info.num_head));
                    ui.end_row();
                });
        });
    })
}

async fn inspect(
    ui: Ui,
    Pack {
        info,
        tokens,
        decoded,
        data,
        predicts,
    }: Pack,
) -> Result<()> {
    let ui_id = uid::Id::<UiId>::new();

    let (rwk, wk) = {
        let num_token = decoded.len();
        let mut rwk = HashMap::new();
        let mut wk = HashMap::new();

        let size = info.num_emb / info.num_head;
        let (tx, rx) = flume::unbounded();
        let total_rk = info.num_layer * info.num_head * num_token * (num_token - 1) / 2;

        for (layer, data) in data.into_iter().enumerate() {
            let info = info.clone();
            let tx = tx.clone();

            let HookDataCpu { k, r, w, .. } = data;
            let k = k.into_iter().map(|x| x.to_vec()).collect_vec();
            let r = r.into_iter().map(|x| x.to_vec()).collect_vec();
            let w = w.into_iter().map(|x| x.to_vec()).collect_vec();

            tokio::task::spawn_blocking(move || {
                for head in 0..info.num_head {
                    let start = size * head;
                    let end = start + size;

                    for source in 0..num_token {
                        let mut k = k[source][start..end].to_vec();

                        for token in source + 1..num_token {
                            let key = HeadKey {
                                layer,
                                head,
                                source,
                                token,
                            };

                            let w = &w[token][start..end];
                            let r = &r[token][start..end];

                            let dot: f32 = k.iter().zip_eq(r).map(|(k, r)| k * r).sum();
                            for (k, w) in k.iter_mut().zip_eq(w) {
                                *k *= w;
                            }

                            let norm = k.iter().map(|x| x * x).sum::<f32>().sqrt();

                            let Ok(_) = tx.send((key, dot, norm)) else {
                                return;
                            };
                        }
                    }
                }
            });
        }

        let (processed_tx, processed_rx) = tokio::sync::watch::channel(0usize);
        let _process_ui = ui.create(move |ctx, _| {
            use egui::{Id, ProgressBar, Window};

            Window::new("Inspector").id(Id::new(ui_id)).show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Processing tokens...");
                    ui.spinner();
                });

                let progress = *processed_rx.borrow() as f32 / total_rk as f32;
                ui.add(ProgressBar::new(progress).text(format!("{:.0}%", progress * 100.0)));
            });
        });

        drop(tx);
        while let Ok((key, dot, norm)) = rx.recv_async().await {
            rwk.insert(key, dot);
            wk.insert(key, norm);
            processed_tx.send_modify(|count| *count += 1);
        }
        (rwk, wk)
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    enum Mode {
        Rwk,
        Wk,
    }

    let num_token = decoded.len();
    let mode = Mutex::new(Mode::Rwk);
    let scale = Mutex::new(0);
    let source = Mutex::new(0usize);
    let layer = Mutex::new(0usize);
    let head = Mutex::new(0usize);

    let (tx, rx) = flume::unbounded();
    let _inspect_ui = ui.create(move |ctx, _| {
        use egui::{
            Color32, Grid, Id, Label, ProgressBar, RichText, ScrollArea, Sense, Slider, Window,
        };

        Window::new("Inspector").id(Id::new(ui_id)).show(ctx, |ui| {
            let mut mode = mode.lock().unwrap();
            let mut scale = scale.lock().unwrap();
            let mut source = source.lock().unwrap();
            let mut layer = layer.lock().unwrap();
            let mut head = head.lock().unwrap();

            ui.horizontal(|ui| {
                ui.radio_value(&mut *mode, Mode::Rwk, "R-W-K dot");
                ui.radio_value(&mut *mode, Mode::Wk, "W-K norm");
            });
            ui.add(Slider::new(&mut *scale, -3..=0).text("Scale"));
            ui.add(Slider::new(&mut *source, 0..=num_token - 1).text("Source Token"));
            ui.add(Slider::new(&mut *layer, 0..=info.num_layer - 1).text("Layer"));
            ui.add(Slider::new(&mut *head, 0..=info.num_head - 1).text("Head"));
            ui.separator();

            ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    for (index, decoded) in decoded.iter().enumerate() {
                        let text = match index.cmp(&source) {
                            Ordering::Less => {
                                let layer = *layer;
                                let head = *head;
                                let token = *source;
                                let source = index;
                                let mode = *mode;
                                let scale = 10.0_f32.powi(*scale);

                                let key = HeadKey {
                                    layer,
                                    head,
                                    source,
                                    token,
                                };
                                if let Some(v) = match mode {
                                    Mode::Rwk => rwk.get(&key),
                                    Mode::Wk => wk.get(&key),
                                } {
                                    let v = (v * scale).clamp(-1.0, 1.0);
                                    let color = if v >= 0.0 {
                                        Color32::LIGHT_RED.gamma_multiply(v)
                                    } else {
                                        Color32::LIGHT_GREEN.gamma_multiply(-v)
                                    };
                                    RichText::new(decoded).color(color)
                                } else {
                                    RichText::new(decoded)
                                }
                            }
                            Ordering::Equal => {
                                let color = Color32::LIGHT_BLUE;
                                RichText::new(decoded).color(color)
                            }
                            Ordering::Greater => {
                                let layer = *layer;
                                let head = *head;
                                let source = *source;
                                let token = index;
                                let mode = *mode;
                                let scale = 10.0_f32.powi(*scale);

                                let key = HeadKey {
                                    layer,
                                    head,
                                    source,
                                    token,
                                };
                                if let Some(v) = match mode {
                                    Mode::Rwk => rwk.get(&key),
                                    Mode::Wk => wk.get(&key),
                                } {
                                    let v = (v * scale).clamp(-1.0, 1.0);
                                    let color = if v >= 0.0 {
                                        Color32::LIGHT_RED.gamma_multiply(v)
                                    } else {
                                        Color32::LIGHT_GREEN.gamma_multiply(-v)
                                    };
                                    RichText::new(decoded).color(color)
                                } else {
                                    RichText::new(decoded)
                                }
                            }
                        };
                        let label = Label::new(text).sense(Sense::click());
                        let response = ui.add(label);
                        if response.clicked() {
                            *source = index;
                        }
                        response.on_hover_ui(|ui| {
                            Grid::new("token").striped(true).show(ui, |ui| {
                                let selected = tokens.get(index + 1).unwrap_or(&0);

                                ui.strong("Rank");
                                ui.strong("Token");
                                ui.strong("Text");
                                ui.strong("Probability");
                                ui.end_row();

                                for (rank, (token, decoded, prob)) in
                                    predicts[index].iter().enumerate()
                                {
                                    let decoded = decoded.replace('\n', "↩");
                                    if *token == *selected {
                                        let color = Color32::LIGHT_BLUE;
                                        ui.colored_label(color, rank.to_string());
                                        ui.colored_label(color, token.to_string());
                                        ui.colored_label(color, decoded);
                                    } else {
                                        ui.label(rank.to_string());
                                        ui.label(token.to_string());
                                        ui.label(decoded);
                                    }
                                    ui.add(ProgressBar::new(*prob).text(format!("{:.4}", prob)));
                                    ui.end_row();
                                }
                            });
                        });
                    }
                });
            });

            ui.separator();
            if ui.button("Back").clicked() {
                let _ = tx.send(());
            }
        });
    });

    rx.recv_async().await?;
    Ok(())
}

async fn load_fonts(ui: Ui) -> Result<()> {
    let path = "assets/fonts/NotoSansCJK-SC/NotoSansCJKsc-Regular.ttf";
    let file = File::open(path).await?;
    let data = unsafe { Mmap::map(&file)? };

    ui.create(move |ctx, _| {
        let data = data.to_vec();
        let mut fonts = egui::FontDefinitions::default();
        fonts
            .font_data
            .insert("noto_sans".into(), egui::FontData::from_owned(data));
        fonts
            .families
            .get_mut(&egui::FontFamily::Proportional)
            .unwrap()
            .push("noto_sans".into());

        ctx.set_fonts(fonts);
    });

    Ok(())
}

#[tokio::main]
async fn main() {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .with_module_level("web_rwkv_inspector", log::LevelFilter::Info)
        .init()
        .unwrap();

    let (app, ui) = App::new();
    let app = Box::new(app);
    tokio::spawn(load_fonts(ui.clone()));
    tokio::spawn(run(ui.clone()));
    tokio::spawn(trace(ui));

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "Web-RWKV Inspector",
        options,
        Box::new(|creation_context| {
            creation_context.egui_ctx.set_theme(egui::Theme::Dark);
            Ok(app)
        }),
    )
    .unwrap();
}
