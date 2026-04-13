use std::time::Instant;

use anyhow::Result;
use rerank::{CrossEncoderReranker, RankedDocument, Reranker, RerankerConfig};
use tantivy::Index;
use tokio;
#[tokio::main]
async fn main() -> Result<()> {
    let query = "How many people live in Berlin?";
    let docs = vec![
        "Berlin is well known for its museums.",
        "A large metropolitan city had a population of 4123456 registered residents
         Reranking is a critical stage in contemporary information retrieval (IR) systems",
        "Berlin Population increased vey much",
        "Berlin Population is 3M.",
    ];
    let config = RerankerConfig {
        model_path: "models/model.onnx".to_string(),
        tokenizer_path: "json/tokenizer.json".to_string(),
    };

    let reranker = CrossEncoderReranker::new(config).unwrap();
    let ranked_docs = reranker.rerank(query, docs, 5).await?;
    println!("ranked_docs: {:?}", ranked_docs);

    Ok(())
}
