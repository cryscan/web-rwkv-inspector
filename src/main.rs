use std::path::PathBuf;

use anyhow::{bail, Result};
use half::f16;
use memmap2::Mmap;
use safetensors::SafeTensors;
use tokio::fs::File;
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    runtime::{
        infer::{InferInput, InferOutput},
        loader::Loader,
        model::{
            AsAny, Build, ContextAutoLimits, ModelBuilder, ModelInfo, ModelRuntime, ModelVersion,
        },
        v6, JobRuntime,
    },
    wgpu,
};

#[derive(Debug, Clone)]
struct Runtime {
    runtime: JobRuntime<InferInput, InferOutput<f16>>,
    model: v6::Model,
    state: v6::State,
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

    let builder = v6::ModelJobBuilder::new(model.clone(), 1);
    let state = builder.state();
    let state = state.as_any().downcast_ref::<v6::State>().unwrap().clone();
    let runtime = JobRuntime::new(builder).await;

    Ok(Runtime {
        runtime,
        model,
        state,
    })
}

#[derive(Debug, Clone)]
enum App {
    Start {
        load_sender: flume::Sender<(PathBuf, flume::Sender<Result<Runtime>>)>,
    },
    Loading {
        path: PathBuf,
        runtime_receiver: flume::Receiver<Result<Runtime>>,
        load_sender: flume::Sender<(PathBuf, flume::Sender<Result<Runtime>>)>,
    },
    Loaded(Runtime),
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.clone() {
                App::Start { load_sender } => {
                    if ui.button("Open file...").clicked() {
                        if let Some(path) = rfd::FileDialog::new().pick_file() {
                            let (runtime_sender, runtime_receiver) = flume::unbounded();
                            let _ = load_sender.send((path.clone(), runtime_sender));
                            *self = App::Loading {
                                path,
                                runtime_receiver,
                                load_sender,
                            };
                        }
                    }
                }
                App::Loading {
                    path,
                    runtime_receiver,
                    load_sender,
                } => {
                    ui.label(format!("Loading model from {}...", path.to_string_lossy()));

                    if let Ok(runtime) = runtime_receiver.try_recv() {
                        match runtime {
                            Ok(runtime) => *self = App::Loaded(runtime),
                            Err(err) => {
                                log::error!("{}", err);
                                *self = App::Start { load_sender };
                            }
                        }
                    }
                }
                App::Loaded(runtime) => {}
            };
        });
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

    let (load_sender, load_receiver) = flume::unbounded();
    let app = App::Start { load_sender };

    tokio::spawn(async move {
        loop {
            let Ok((path, sender)) = load_receiver.recv_async().await else {
                continue;
            };
            let runtime = load_runtime(path).await;
            let _ = sender.send(runtime);
        }
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native("Web-RWKV Inspector", options, Box::new(|_| Box::new(app))).unwrap();
}
