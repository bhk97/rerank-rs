use crate::errors::RerankerError;
use anyhow::{Result, anyhow};
use ort::session::{Session, builder::GraphOptimizationLevel};
pub(crate) fn load_model(path: &str) -> Result<Session, RerankerError> {
    let mut builder = Session::builder().map_err(|_| RerankerError::ModelLoad)?;

    builder = builder
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|_| RerankerError::ModelLoad)?;

    builder = builder
        .with_intra_threads(12)
        .map_err(|_| RerankerError::ModelLoad)?;

    let model = builder
        .commit_from_file(path)
        .map_err(|_| RerankerError::ModelLoad)?;
    Ok(model)
}
