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

#[derive(Debug)]
pub struct RankedDocument {
    pub index: usize,
    pub text: String,
    pub score: f32,
}

pub struct RerankerConfig {
    pub model_path: String,
    pub tokenizer_path: String,
}

pub struct CrossEncoderReranker {
    pub model: Arc<Mutex<Session>>,
    pub tokenizer: Tokenizer,
}

#[derive(Debug, Clone)]
pub struct WindowedPair {
    pub query_doc_pair: (String, String),
    pub doc_index: usize,
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
