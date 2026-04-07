use crate::errors::RerankerError;
use anyhow::Result;
use ort::session::Session;
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

pub struct ModelAndTokenizer {
    pub model: Session,
    pub tokenizer: tokenizers::Tokenizer,
}
