use std::{
    path::PathBuf,
    sync::{atomic::AtomicUsize, Arc},
};

use anyhow::{bail, Result};
use half::f16;
use memmap2::Mmap;
use safetensors::SafeTensors;
use tokio::{
    fs::File,
    sync::watch::{Receiver, Sender},
};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    runtime::{
        infer::{InferInput, InferOutput},
        loader::Loader,
        model::{Build, ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion},
        v6, JobRuntime,
    },
    tensor::{kind::ReadWrite, TensorGpu},
    wgpu,
};

const MAX_INPUT_LEN: usize = 1024;

#[derive(Debug, Clone)]
struct Runtime {
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

async fn load_runtime(path: PathBuf) -> Result<Runtime> {
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
            let num_token = MAX_INPUT_LEN;
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
        runtime,
        model,
        frame,
    })
}

type BoxUi = Box<dyn Fn(&egui::Context, &mut eframe::Frame) + Send + Sync>;

struct App(Receiver<BoxUi>);

impl App {
    fn new<F>(f: F) -> (Self, Sender<BoxUi>)
    where
        F: Fn(&egui::Context, &mut eframe::Frame) + Send + Sync + 'static,
    {
        let f: Box<dyn Fn(&egui::Context, &mut eframe::Frame) + Send + Sync> = Box::new(f);
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

async fn run(ui: Sender<BoxUi>) -> Result<()> {
    'main: loop {
        // choose model file
        let runtime = {
            let (tx, rx) = flume::unbounded();
            ui.send(Box::new(move |ctx, _| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    if ui.button("Open file...").clicked() {
                        if let Some(path) = rfd::FileDialog::new().pick_file() {
                            let _ = tx.send(path);
                        }
                    }
                });
            }))?;

            let Ok(path) = rx.recv_async().await else {
                continue 'main;
            };

            {
                let path = path.clone();
                ui.send(Box::new(move |ctx, _| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        ui.label(format!("Loading model from {}...", path.to_string_lossy()));
                    });
                }))?
            };

            match load_runtime(path).await {
                Ok(runtime) => runtime,
                Err(err) => {
                    let (tx, rx) = flume::unbounded();
                    ui.send(Box::new(move |ctx, _| {
                        egui::CentralPanel::default().show(ctx, |ui| {
                            ui.heading("Error");
                            ui.label(err.to_string());
                            if ui.button("Ok").clicked() {
                                let _ = tx.send(());
                            }
                        });
                    }))?;

                    let _ = rx.recv_async().await;
                    continue 'main;
                }
            }
        };
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

    let (app, sender) = App::new(|_, _| {});
    let app = Box::new(app);
    tokio::spawn(run(sender));

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native("Web-RWKV Inspector", options, Box::new(|_| app)).unwrap();
}
