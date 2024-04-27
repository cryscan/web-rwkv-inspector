use std::{
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
        model::{Build, ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion},
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
}

#[derive(Debug, Default, Clone)]
struct HookDataCpu {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    r: Vec<Vec<f32>>,
    w: Vec<Vec<f32>>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct HeadKey {
    layer: usize,
    head: usize,
    token: usize,
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

async fn load_runtime(path: PathBuf) -> Result<Runtime> {
    let tokenizer = Arc::new(load_tokenizer().await?);

    let file = File::open(path).await?;
    let data = unsafe { Mmap::map(&file)? };

    let model = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model)?;
    log::info!("{:#?}", info);

    if !matches!(info.version, ModelVersion::V6) {
        bail!("only supports V6 models");
    }

    let context = create_context(&info).await?;
    log::info!("{:#?}", context.adapter.get_info());

    let builder = ModelBuilder::new(&context, model);
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

struct UiHandle(Arc<()>);

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

        match load_runtime(path).await {
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

        loop {
            let select = {
                enum TokenSelection {
                    Back,
                    Token(usize),
                    Select,
                }

                let decoded = decoded.clone();
                let (tx, rx) = flume::unbounded();
                let (select_tx, select_rx) = tokio::sync::watch::channel(usize::MAX);
                let _ui_input = ui.create(move |ctx, _| {
                    egui::Window::new("Tokens")
                        .max_height(400.0)
                        .show(ctx, |ui| {
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                ui.horizontal_wrapped(|ui| {
                                    for (index, token) in decoded.iter().enumerate() {
                                        let button = if index == *select_rx.borrow() {
                                            let token = token.replace('\n', "↩");
                                            ui.small_button(
                                                egui::RichText::new(token)
                                                    .color(egui::Color32::LIGHT_RED),
                                            )
                                        } else {
                                            let token = token.replace('\n', "↩");
                                            ui.small_button(token)
                                        };
                                        if button.clicked() {
                                            let _ = tx.send(TokenSelection::Token(index));
                                        }
                                        if token.contains('\n') {
                                            ui.end_row();
                                        }
                                    }
                                });
                            });
                            ui.separator();
                            ui.horizontal(|ui| {
                                if ui.button("Back").clicked() {
                                    let _ = tx.send(TokenSelection::Back);
                                }
                                if *select_rx.borrow() != usize::MAX
                                    && ui.button("Select").clicked()
                                {
                                    let _ = tx.send(TokenSelection::Select);
                                }
                            });
                        });
                });

                'select: loop {
                    match rx.recv_async().await? {
                        TokenSelection::Back => continue 'input,
                        TokenSelection::Token(index) => select_tx.send(index)?,
                        TokenSelection::Select => break 'select *select_tx.borrow(),
                    }
                }
            };

            let data = data.clone();
            let info = runtime.model.info.clone();
            let handle = tokio::task::spawn_blocking(move || {
                let size = info.num_emb / info.num_head;
                let mut kv = HashMap::new();
                let mut rkv = HashMap::new();

                for (layer, data) in data.into_iter().enumerate() {
                    for head in 0..info.num_head {
                        let token = select;
                        let start = size * head;
                        let end = start + size;
                        let k = &data.k[token][start..end];
                        let v = &data.v[token][start..end];
                        let mut tensor = Vec::with_capacity(size * size);
                        for (v, k) in v.iter().cartesian_product(k.iter()) {
                            tensor.push(k * v);
                        }
                        kv.insert(HeadKey { layer, head, token }, tensor.clone());
                        for token in (token + 1)..num_token {
                            let w = &data.w[token][start..end];
                            let r = &data.r[token][start..end];
                            for (j, i) in (0..size).cartesian_product(0..size) {
                                tensor[j * size + i] *= w[i];
                            }
                            kv.insert(HeadKey { layer, head, token }, tensor.clone());

                            let mut tensor = tensor.clone();
                            for (j, i) in (0..size).cartesian_product(0..size) {
                                tensor[j * size + i] *= r[i];
                            }
                            rkv.insert(HeadKey { layer, head, token }, tensor);
                        }
                    }
                }

                (kv, rkv)
            });

            let (kv, rkv) = {
                let _inspect_ui = ui.create(|ctx, _| {
                    egui::Window::new("Inspect").show(ctx, |ui| {
                        ui.spinner();
                    });
                });
                handle.await?
            };

            let decoded = decoded.clone();
            let (tx, rx) = flume::unbounded();
            let kv_textures = Mutex::new(HashMap::<HeadKey, egui::TextureHandle>::new());
            let rkv_textures = Mutex::new(HashMap::<HeadKey, egui::TextureHandle>::new());
            let token = Mutex::new(select);
            let show_rkv = Mutex::new(false);
            let _inspect_ui = ui.create(move |ctx, _| {
                let mut token = token.lock().unwrap();
                let mut show_rkv = show_rkv.lock().unwrap();

                let size = info.num_emb / info.num_head;

                egui::Window::new("Inspect").show(ctx, |ui| {
                    if ui.button("Back").clicked() {
                        let _ = tx.send(());
                    }

                    ui.separator();
                    ui.checkbox(&mut show_rkv, "Show R-Queried W-decayed KV");

                    // let slider: egui::Slider<'_> =
                    //     egui::Slider::new(&mut *scale, -12..=0).text("Scale");
                    // ui.add(slider);

                    let slider =
                        egui::Slider::new(&mut *token, select..=num_token - 1).text("Token");
                    ui.add(slider);
                    ui.label(decoded[*token].replace('\n', "↩"));

                    ui.separator();
                    egui::ScrollArea::both().show(ui, |ui| {
                        egui::Grid::new("image").spacing([8.0, 4.0]).show(ui, |ui| {
                            ui.label("");
                            for head in 0..info.num_head {
                                ui.label(format!("Head {head}"));
                            }
                            ui.end_row();

                            for layer in 0..info.num_layer {
                                ui.label(format!("Layer {layer}"));

                                for head in 0..info.num_head {
                                    let token = *token;
                                    let scale = 1.0;
                                    let key = HeadKey { layer, head, token };
                                    let mut image = egui::epaint::ColorImage::new(
                                        [size, size],
                                        egui::Color32::BLACK,
                                    );
                                    if let Some(kv) = match *show_rkv {
                                        true => rkv.get(&key),
                                        false => kv.get(&key),
                                    } {
                                        let norm = kv
                                            .iter()
                                            .map(|x| x * scale)
                                            .map(|x| x.clamp(-1.0, 1.0));
                                        for (pixel, norm) in image.pixels.iter_mut().zip_eq(norm) {
                                            if norm >= 0.0 {
                                                pixel[0] = (norm * 255.0) as u8;
                                            } else {
                                                pixel[1] = (-norm * 255.0) as u8;
                                            }
                                        }
                                    }
                                    let mut textures = match *show_rkv {
                                        true => rkv_textures.lock().unwrap(),
                                        false => kv_textures.lock().unwrap(),
                                    };
                                    let texture = match textures.get(&key) {
                                        Some(texture) => texture.clone(),
                                        None => {
                                            let texture = ctx.load_texture(
                                                format!("kv_{:?}", key),
                                                image,
                                                Default::default(),
                                            );
                                            textures.insert(key, texture.clone());
                                            texture
                                        }
                                    };
                                    ui.image((texture.id(), texture.size_vec2()));
                                }
                                ui.end_row();
                            }
                        });
                    });
                });
            });
            rx.recv_async().await?;
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
