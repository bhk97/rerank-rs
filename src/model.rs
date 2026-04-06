use anyhow::{Result, anyhow};
use ort::session::{Session, builder::GraphOptimizationLevel};

pub fn load_model() -> Result<Session> {
    let mut builder = Session::builder().map_err(|e| anyhow!("{:?}", e))?;

    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("{:?}", e))?;

    builder = builder
        .with_intra_threads(4)
        .map_err(|e| anyhow!("{:?}", e))?;

    let mut model = builder
        .commit_from_file("models/model.onnx")
        .map_err(|e| anyhow!("{:?}", e))?;
    Ok(model)
}
