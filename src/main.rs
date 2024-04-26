use std::{
    path::PathBuf,
    sync::{atomic::AtomicUsize, Arc, Mutex},
};

use anyhow::{bail, Result};
use half::f16;
use memmap2::Mmap;
use safetensors::SafeTensors;
use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
    sync::watch::{Receiver, Sender},
};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    runtime::{
        infer::{InferInput, InferInputBatch, InferOption, InferOutput},
        loader::Loader,
        model::{Build, ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion},
        v6, JobRuntime, Submission,
    },
    tensor::{kind::ReadWrite, TensorGpu},
    tokenizer::Tokenizer,
    wgpu,
};

const MAX_INPUT_TOKENS: usize = 4096;

#[derive(Debug, Clone)]
struct Runtime {
    tokenizer: Arc<Tokenizer>,
    runtime: JobRuntime<InferInput, InferOutput<f16>>,
    model: v6::Model,
    frame: HookFrame,
}

#[derive(Debug, Clone)]
struct HookFrame {
    index: Arc<AtomicUsize>,
    layers: Vec<LayerHookFrame>,
}

#[derive(Debug, Clone)]
struct LayerHookFrame {
    k: TensorGpu<f32, ReadWrite>,
    v: TensorGpu<f32, ReadWrite>,
    r: TensorGpu<f32, ReadWrite>,
    w: TensorGpu<f32, ReadWrite>,
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

    let layers = (0..model.info.num_layer)
        .map(|_| {
            let context = &model.context;
            let num_emb = model.info.num_emb;
            let num_token = MAX_INPUT_TOKENS;
            let shape = [num_emb, num_token, 1, 1];
            LayerHookFrame {
                k: context.tensor_init(shape),
                v: context.tensor_init(shape),
                r: context.tensor_init(shape),
                w: context.tensor_init(shape),
            }
        })
        .collect();
    let frame = HookFrame {
        index: Arc::new(AtomicUsize::new(0)),
        layers,
    };

    let builder = v6::ModelJobBuilder::new(model.clone(), 1);
    let runtime = JobRuntime::new(builder).await;

    Ok(Runtime {
        tokenizer,
        runtime,
        model,
        frame,
    })
}

type ArcUi = Arc<dyn Fn(&egui::Context, &mut eframe::Frame) + Send + Sync>;

struct App(Receiver<ArcUi>);

impl App {
    fn new() -> (Self, Sender<ArcUi>) {
        let f: Arc<dyn Fn(&egui::Context, &mut eframe::Frame) + Send + Sync> = Arc::new(|_, _| ());
        let (sender, receiver) = tokio::sync::watch::channel(f);
        (Self(receiver), sender)
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let f = self.0.borrow();
        f(ctx, frame);
    }
}

async fn run(ui: Sender<ArcUi>) -> Result<()> {
    let runtime = 'load: loop {
        let (tx, rx) = flume::unbounded();
        ui.send(Arc::new(move |ctx, _| {
            egui::Window::new("Load").title_bar(false).show(ctx, |ui| {
                ui.label("Choose a model file.");
                if ui.button("Open...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("SafeTensors", &["st"])
                        .pick_file()
                    {
                        let _ = tx.send(path);
                    }
                }
            });
        }))?;

        let Ok(path) = rx.recv_async().await else {
            continue 'load;
        };

        {
            let path = path.clone();
            ui.send(Arc::new(move |ctx, _| {
                egui::Window::new("Load").title_bar(false).show(ctx, |ui| {
                    ui.label(format!("Loading model from {}...", path.to_string_lossy()));
                });
            }))?
        };

        match load_runtime(path).await {
            Ok(runtime) => break 'load runtime,
            Err(err) => {
                let (tx, rx) = flume::unbounded();
                ui.send(Arc::new(move |ctx, _| {
                    egui::Window::new("Load").title_bar(false).show(ctx, |ui| {
                        ui.label(format!("Error: {}", err));
                        if ui.button("Ok").clicked() {
                            let _ = tx.send(());
                        }
                    });
                }))?;

                let _ = rx.recv_async().await;
                continue 'load;
            }
        }
    };

    let ui_info: ArcUi = {
        let info = runtime.model.info.clone();
        Arc::new(move |ctx, _| {
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
        let text = Mutex::new(String::new());

        let ui_info = ui_info.clone();
        ui.send(Arc::new(move |ctx, frame| {
            ui_info(ctx, frame);

            egui::Window::new("Input").show(ctx, |ui| {
                let mut text = text.lock().unwrap();
                let text = &mut *text;
                ui.text_edit_multiline(text);

                if ui.button("Submit").clicked() {
                    let _ = tx.send(text.clone());
                }
            });
        }))?;

        let Ok(input) = rx.recv_async().await else {
            continue 'input;
        };

        let Ok(mut tokens) = runtime.tokenizer.encode(&input.into_bytes()) else {
            continue 'input;
        };
        tokens.truncate(MAX_INPUT_TOKENS);

        let input = InferInput::new(
            vec![InferInputBatch {
                tokens,
                option: InferOption::Full,
            }],
            128,
        );
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let submission = Submission { input, sender };
        runtime.runtime.send(submission).await?;
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
