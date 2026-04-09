use thiserror::Error;

#[derive(Debug, Error)]
pub enum RerankerError {
    #[error("Failed to Load Model")]
    ModelLoad,

    #[error("Failed to Load Tokenizer")]
    FileLoad,

    #[error("Problem with your model file")]
    Onnx(#[from] ort::error::Error),

    #[error("Error in tokenizer calculation")]
    Tokenizer(#[from] Box<dyn std::error::Error + Send + Sync>),
    /// Invalid Input Error
    #[error("Invalid input")]
    InvalidInput,

    #[error("Shape Error In Predicting Scores")]
    ShapeError,
}
