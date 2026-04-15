use thiserror::Error;

#[derive(Debug, Error)]
pub enum RerankerError {
    /// Error in loading the model
    #[error("Failed to Load Model")]
    ModelLoad,

    /// Error in loading the tokenizer
    #[error("Failed to Load Tokenizer")]
    FileLoad,

    /// Error in the ONNX model
    #[error("Problem with your model file")]
    Onnx(#[from] ort::error::Error),

    /// Error in the tokenizer calculation
    #[error("Error in tokenizer calculation")]
    Tokenizer(#[from] Box<dyn std::error::Error + Send + Sync>),
    /// Invalid Input Error
    #[error("Invalid input")]
    InvalidInput,

    /// Shape Error In Predicting Scores
    #[error("Shape Error In Predicting Scores")]
    ShapeError,

    ///Building Tokens error
    #[error("Building Tokens error")]
    BuildingTokensError(String),
}
