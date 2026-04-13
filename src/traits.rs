use crate::errors::RerankerError;
use anyhow::Result;
use ort::session::Session;
use ort::value::{TensorValueType, Value};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
pub trait Reranker {
    async fn rerank(
        &self,
        query: &str,
        docs: Vec<&str>,
        top_n: usize,
    ) -> Result<Vec<RankedDocument>, RerankerError>;
}
/// Struct to hold the ranked documents
#[derive(Debug)]
pub struct RankedDocument {
    /// Index of the document in the original list
    pub index: usize,
    /// Text of the document
    pub text: String,
    /// Score of the document
    pub score: f32,
}

/// Struct to hold the reranker configuration
pub struct RerankerConfig {
    /// Path to the model file
    pub model_path: String,
    /// Path to the tokenizer file
    pub tokenizer_path: String,
}

/// Struct to hold the cross encoder reranker
pub struct CrossEncoderReranker {
    /// Model and tokenizer
    pub model: Arc<Mutex<Session>>,
    pub tokenizer: Tokenizer,
}

#[derive(Debug, Clone)]
pub struct WindowedPair {
    #[allow(dead_code)]
    pub query_doc_pair: (String, String),
    pub doc_index: usize,
    #[allow(dead_code)]
    pub window_index: usize,
    pub token_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub type_ids: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct TokenizerResult {
    pub tensor_values: Vec<Value<TensorValueType<i64>>>,
    pub windowed_pairs: Vec<WindowedPair>,
}

#[derive(Debug, Clone)]
pub struct BucketResult {
    pub res: Vec<TokenizerResult>,
}
