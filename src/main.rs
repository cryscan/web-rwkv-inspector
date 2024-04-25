use std::{
    path::PathBuf,
    sync::{Arc, RwLock},
};

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

struct App(Arc<RwLock<Box<dyn Fn(&egui::Context, &mut eframe::Frame) + Send + Sync>>>);

impl std::ops::Deref for App {
    type Target = Arc<RwLock<Box<dyn Fn(&egui::Context, &mut eframe::Frame) + Send + Sync>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Default for App {
    fn default() -> Self {
        let ui = |_: &egui::Context, _: &mut eframe::Frame| {};
        let ui: Arc<RwLock<Box<dyn Fn(&egui::Context, &mut eframe::Frame) + Send + Sync>>> =
            Arc::new(RwLock::new(Box::new(ui)));
        Self(ui)
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let ui = self.read().unwrap();
        ui(ctx, frame);
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

    let app = Box::<App>::default();
    let ui = app.0.clone();

    let run = async move {
        'main: loop {
            // choose model file
            let runtime = {
                let (sender, receiver) = flume::unbounded();
                {
                    let mut ui = ui.write().unwrap();
                    *ui = Box::new(move |ctx: &egui::Context, _: &mut eframe::Frame| {
                        egui::CentralPanel::default().show(ctx, |ui| {
                            if ui.button("Open file...").clicked() {
                                if let Some(path) = rfd::FileDialog::new().pick_file() {
                                    let _ = sender.send(path);
                                }
                            }
                        });
                    });
                }

                let Ok(path) = receiver.recv_async().await else {
                    continue 'main;
                };

                {
                    let path = path.clone();
                    let mut ui = ui.write().unwrap();
                    *ui = Box::new(move |ctx: &egui::Context, _: &mut eframe::Frame| {
                        egui::CentralPanel::default().show(ctx, |ui| {
                            ui.label(format!("Loading model from {}...", path.to_string_lossy()));
                        });
                    });
                }

                let Ok(runtime) = load_runtime(path).await else {
                    continue 'main;
                };
                runtime
            };
        }
    };
    tokio::spawn(run);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native("Web-RWKV Inspector", options, Box::new(|_| app)).unwrap();
}
