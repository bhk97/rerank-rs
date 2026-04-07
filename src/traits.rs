use anyhow::Result;

pub trait Reranker {
    async fn rerank(
        &self,
        query: &str,
        docs: Vec<&str>,
        top_n: usize,
    ) -> Result<Vec<RankedDocument>>;
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
    pub config: RerankerConfig,
}
