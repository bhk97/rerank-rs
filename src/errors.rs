use thiserror::Error;
//we need to remove the anyhow and use this errors in phase 2 or 3
#[derive(Error, Debug)]
pub enum RerankerError {
    #[error("Model loading failed: {0}")]
    ModelError(#[from] ort::OrtError),
    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] tokenizers::Error),
    #[error("Input error: {0}")]
    InputError(#[from] anyhow::Error),
    #[error("Output error: {0}")]
    OutputError(#[from] anyhow::Error),
}
