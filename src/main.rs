use anyhow::Result;
mod model;
mod reranker;
mod tokenizer;
mod traits;
use crate::traits::{CrossEncoderReranker, Reranker, RerankerConfig};
use tokio;
mod errors;
#[tokio::main]
async fn main() -> Result<()> {
    let query = "How many people live in Berlin?";
    let docs = vec![
        "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
        "Berlin is well known for its museums.",
        "Berlin Population increased vey much",
        "Berlin Population is 3M.",
    ];
    let config = RerankerConfig {
        model_path: "models/model.onnx".to_string(),
        tokenizer_path: "json/tokenizer.json".to_string(),
    };
    let reranker = CrossEncoderReranker::new(config);
    let doc_sorted = reranker.rerank(query, docs, 1).await?;
    println!("docs: {:?}", doc_sorted);

    Ok(())
}
