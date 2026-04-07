use anyhow::{Result, anyhow};
use ort::session::{Session, builder::GraphOptimizationLevel};

pub fn load_model(path: &str) -> Result<Session> {
    let mut builder = Session::builder().map_err(|e| anyhow!("{:?}", e))?;

    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("{:?}", e))?;

    builder = builder
        .with_intra_threads(4)
        .map_err(|e| anyhow!("{:?}", e))?;

    let model = builder
        .commit_from_file(path)
        .map_err(|e| anyhow!("{:?}", e))?;
    Ok(model)
}
