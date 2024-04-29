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
use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    num::{CoHom, Float},
    runtime::{
        infer::{InferInput, InferInputBatch, InferOption, InferOutput},
        loader::Loader,
        model::{Build, ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant},
        v6, JobRuntime, Submission,
    },
    tensor::{kind::ReadWrite, ops::TensorOp, TensorCpu, TensorGpu, TensorShape},
    tokenizer::Tokenizer,
    wgpu,
};

const MAX_INPUT_TOKENS: usize = 4096;
const TOKEN_CHUNK_SIZE: usize = 128;

#[derive(Debug, Clone)]
struct Runtime {
    tokenizer: Arc<Tokenizer>,
    runtime: JobRuntime<InferInput, InferOutput<f16>>,
    model: v6::Model,
    data: Vec<HookDataGpu>,
}

#[derive(Debug, Clone)]
struct HookDataGpu {
    k: TensorGpu<f32, ReadWrite>,
    v: TensorGpu<f32, ReadWrite>,
    r: TensorGpu<f32, ReadWrite>,
    w: TensorGpu<f32, ReadWrite>,
    g: TensorGpu<f32, ReadWrite>,
}

#[derive(Debug, Default, Clone)]
struct HookDataCpu {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    r: Vec<Vec<f32>>,
    w: Vec<Vec<f32>>,
    g: Vec<Vec<f32>>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct HeadKey {
    layer: usize,
    head: usize,
    source: usize,
    token: usize,
    gated: bool,
}

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = Instance::new();
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

async fn load_runtime(ui: Ui, path: PathBuf) -> Result<Runtime> {
    let tokenizer = Arc::new(load_tokenizer().await?);

    let file = File::open(path).await?;
    let data = unsafe { Mmap::map(&file)? };

    let model = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model)?;
    log::info!("{:#?}", info);

    if !matches!(info.version, ModelVersion::V6) {
        bail!("only supports V6 models");
    }

    let quant = {
        let (tx, rx) = flume::unbounded();
        let quant = Mutex::new(0usize);
        let quant_nf4 = Mutex::new(0usize);
        let _load_ui = ui.create(move |ctx, _| {
            let mut quant = quant.lock().unwrap();
            let mut quant_nf4 = quant_nf4.lock().unwrap();

            egui::Window::new("Configure").show(ctx, |ui| {
                ui.add(egui::Slider::new(&mut *quant, 0..=info.num_layer).text("Int8 Layers"));
                ui.add(egui::Slider::new(&mut *quant_nf4, 0..=info.num_layer).text("NF4 Layers"));

                ui.separator();
                if ui.button("Load").clicked() {
                    let quant = (0..*quant)
                        .map(|layer| (layer, Quant::Int8))
                        .chain((0..*quant_nf4).map(|layer| (layer, Quant::NF4)))
                        .collect();
                    let _ = tx.send(quant);
                }
            });
        });
        rx.recv_async().await?
    };

    let context = create_context(&info).await?;
    log::info!("{:#?}", context.adapter.get_info());

    let builder = ModelBuilder::new(&context, model).quant(quant);
    let model = Build::<v6::Model>::build(builder).await?;

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
                g: context.tensor_init(shape),
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
                    TensorOp::blit(
                        frame.buffer.att_g.view(.., .., .., ..)?,
                        data.g.view(.., ..num_token, .., ..)?,
                    )?,
                ];

                Ok(TensorOp::List(ops))
            }),
        );
    }

    let builder = v6::ModelJobBuilder::new_with_hooks(model.clone(), 1, hooks);
    let runtime = JobRuntime::new(builder).await;

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

        let mut retain = vec![];
        for ui in self.ui.drain(..) {
            if ui.1.upgrade().is_some() {
                (ui.0)(ctx, frame);
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
                egui::Window::new("Load").title_bar(false).show(ctx, |ui| {
                    ui.label("Choose a model file.");
                    ui.add_space(4.0);
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

        match load_runtime(ui.clone(), path).await {
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

    let _ui_info = {
        let info = runtime.model.info.clone();
        ui.create(move |ctx, _| {
            egui::Window::new("Info").show(ctx, |ui| {
                egui::Grid::new("grid")
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
    };

    'input: loop {
        let (tx, rx) = flume::unbounded();
        let (input_tx, input_rx) = tokio::sync::watch::channel(true);
        let text = Mutex::new(String::new());

        let ui_input = ui.create(move |ctx, _| {
            egui::Window::new("Input")
                .max_height(400.0)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
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
        let num_token = tokens.len();

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

        let mut inference = Some(InferInput::new(
            vec![InferInputBatch {
                tokens,
                option: InferOption::Full,
            }],
            128,
        ));

        let mut data = vec![HookDataCpu::default(); runtime.model.info.num_layer];
        'prefill: loop {
            let input = inference.take().unwrap();
            let pre_num_token = input.batches[0].tokens.len();

            let (sender, receiver) = tokio::sync::oneshot::channel();
            let submission = Submission { input, sender };
            let _ = runtime.runtime.send(submission).await;

            let (input, InferOutput(batches)) = receiver.await?;
            let post_num_token = input.batches[0].tokens.len();
            inference = Some(input);

            fn split<F: Float>(x: TensorCpu<F>) -> Vec<Vec<f32>> {
                x.split(1)
                    .expect("split batch")
                    .into_iter()
                    .map(|x| x.map(|&x| CoHom::co_hom(x)).to_vec())
                    .collect()
            }

            let _batch = batches.into_iter().next().unwrap();
            let num_token = pre_num_token - post_num_token;

            for (cpu, gpu) in data.iter_mut().zip_eq(runtime.data.iter()) {
                let mut k = split(gpu.k.back().await);
                let mut v = split(gpu.v.back().await);
                let mut r = split(gpu.r.back().await);
                let mut w = split(gpu.w.back().await);
                let mut g = split(gpu.g.back().await);
                k.truncate(num_token);
                v.truncate(num_token);
                r.truncate(num_token);
                w.truncate(num_token);
                g.truncate(num_token);
                cpu.k.append(&mut k);
                cpu.v.append(&mut v);
                cpu.r.append(&mut r);
                cpu.w.append(&mut w);
                cpu.g.append(&mut g);
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

        let rk = {
            let info = runtime.model.info.clone();
            let data = Arc::new(data);
            let mut rk = HashMap::new();

            let size = info.num_emb / info.num_head;
            let (tx, rx) = flume::unbounded();
            let total_rk = info.num_layer * info.num_head * num_token * (num_token - 1);

            for layer in 0..info.num_layer {
                let info = runtime.model.info.clone();
                let data = data.clone();
                let tx = tx.clone();

                tokio::task::spawn_blocking(move || {
                    for head in 0..info.num_head {
                        let start = size * head;
                        let end = start + size;

                        for source in 0..num_token {
                            let mut k = data[layer].k[source][start..end].to_vec();
                            let v = &data[layer].v[source][start..end];

                            for token in source + 1..num_token {
                                let w = &data[layer].w[token][start..end];
                                let r = &data[layer].r[token][start..end];
                                for (k, w) in k.iter_mut().zip_eq(w) {
                                    *k *= w;
                                }

                                let dot: f32 = k.iter().zip_eq(r).map(|(k, r)| k * r).sum();
                                let key = HeadKey {
                                    layer,
                                    head,
                                    source,
                                    token,
                                    gated: false,
                                };

                                let Ok(_) = tx.send((key, dot)) else {
                                    return;
                                };

                                let g = &data[layer].g[token][start..end];
                                let g = g.iter().map(|x| x / (1.0 + (-x).exp()));

                                let out: f32 = g.zip_eq(v).map(|(g, v)| (g * v).abs()).sum();
                                let out = out / size as f32;
                                let key = HeadKey {
                                    layer,
                                    head,
                                    source,
                                    token,
                                    gated: true,
                                };

                                let Ok(_) = tx.send((key, out * dot)) else {
                                    return;
                                };
                            }
                        }
                    }
                });
            }

            let (processed_tx, processed_rx) = tokio::sync::watch::channel(0usize);
            let _process_ui = ui.create(move |ctx, _| {
                egui::Window::new("Inspector").show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Processing tokens...");
                        ui.spinner();
                    });

                    let progress = *processed_rx.borrow() as f32 / total_rk as f32;
                    ui.add(egui::ProgressBar::new(progress));
                });
            });

            while *processed_tx.borrow() < total_rk {
                let (key, value) = rx.recv_async().await?;
                rk.insert(key, value);
                processed_tx.send_modify(|count| *count += 1);
            }

            rk
        };

        {
            enum InspectorEvent {
                Back,
            }

            let info = runtime.model.info.clone();
            let decoded = decoded.clone();
            let gated = Mutex::new(false);
            let scale = Mutex::new(0);
            let source = Mutex::new(0usize);
            let layer = Mutex::new(0usize);
            let head = Mutex::new(0usize);

            let (tx, rx) = flume::unbounded();
            let _inspect_ui = ui.create(move |ctx, _| {
                egui::Window::new("Inspector").show(ctx, |ui| {
                    let mut gated = gated.lock().unwrap();
                    let mut scale = scale.lock().unwrap();
                    let mut source = source.lock().unwrap();
                    let mut layer = layer.lock().unwrap();
                    let mut head = head.lock().unwrap();

                    ui.add(egui::Checkbox::new(&mut *gated, "Gated"));
                    ui.add(egui::Slider::new(&mut *scale, -6..=0).text("Scale"));
                    ui.add(egui::Slider::new(&mut *source, 0..=num_token - 1).text("Source Token"));
                    ui.add(egui::Slider::new(&mut *layer, 0..=info.num_layer - 1).text("Layer"));
                    ui.add(egui::Slider::new(&mut *head, 0..=info.num_head - 1).text("Head"));
                    ui.separator();

                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            for (index, word) in decoded.iter().enumerate() {
                                let text = match index.cmp(&source) {
                                    Ordering::Less => {
                                        let layer = *layer;
                                        let head = *head;
                                        let token = *source;
                                        let source = index;
                                        let gated = *gated;
                                        let scale = 10.0_f32.powi(*scale);

                                        let key = HeadKey {
                                            layer,
                                            head,
                                            source,
                                            token,
                                            gated,
                                        };
                                        if let Some(rk) = rk.get(&key) {
                                            let rk = (rk * scale).clamp(-1.0, 1.0);
                                            let color = if rk >= 0.0 {
                                                egui::Color32::LIGHT_RED.gamma_multiply(rk)
                                            } else {
                                                egui::Color32::LIGHT_GREEN.gamma_multiply(-rk)
                                            };
                                            egui::RichText::new(word).color(color)
                                        } else {
                                            egui::RichText::new(word)
                                        }
                                    }
                                    Ordering::Equal => {
                                        let color = egui::Color32::LIGHT_BLUE;
                                        egui::RichText::new(word).color(color)
                                    }
                                    Ordering::Greater => {
                                        let layer = *layer;
                                        let head = *head;
                                        let source = *source;
                                        let token = index;
                                        let gated = *gated;
                                        let scale = 10.0_f32.powi(*scale);

                                        let key = HeadKey {
                                            layer,
                                            head,
                                            source,
                                            token,
                                            gated,
                                        };
                                        if let Some(rk) = rk.get(&key) {
                                            let rk = (rk * scale).clamp(-1.0, 1.0);
                                            let color = if rk >= 0.0 {
                                                egui::Color32::LIGHT_RED.gamma_multiply(rk)
                                            } else {
                                                egui::Color32::LIGHT_GREEN.gamma_multiply(-rk)
                                            };
                                            egui::RichText::new(word).color(color)
                                        } else {
                                            egui::RichText::new(word)
                                        }
                                    }
                                };
                                let label = egui::Label::new(text).sense(egui::Sense::click());
                                if ui.add(label).clicked() {
                                    *source = index;
                                }
                            }
                        });
                    });

                    ui.separator();
                    if ui.button("Back").clicked() {
                        let _ = tx.send(InspectorEvent::Back);
                    }
                });
            });

            match rx.recv_async().await? {
                InspectorEvent::Back => continue 'input,
            }
        }
    }
}

#[tokio::main]
async fn main() {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .with_module_level("web_rwkv_inspector", log::LevelFilter::Info)
        .init()
        .unwrap();

    let (app, sender) = App::new();
    let app = Box::new(app);
    tokio::spawn(run(sender));

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_drag_and_drop(true),
        follow_system_theme: false,
        default_theme: eframe::Theme::Dark,
        ..Default::default()
    };
    eframe::run_native("Web-RWKV Inspector", options, Box::new(|_| app)).unwrap();
}
