pub mod errors;
mod model;
mod reranker;
mod tokenizer;
mod traits;
pub use traits::{CrossEncoderReranker, RankedDocument, Reranker, RerankerConfig};

// ! # rerank-rs
// !
// ! Local cross-encoder reranking for Rust.
// ! Zero Python, zero API calls, zero sidecar processes.
// !
// ! ## Quick Start
// ! ```rust,no_run
// ! use rerank_rs::{CrossEncoderReranker, RerankerConfig};
// !
// ! let reranker = CrossEncoderReranker::new(RerankerConfig {
// !     model_path: "model.onnx".into(),
// !     tokenizer_path: "tokenizer.json".into(),
// ! })?;
// ! ```
