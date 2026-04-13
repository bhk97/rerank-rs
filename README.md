# rerank-rs

Local cross-encoder reranking for Rust. No API calls, no Python sidecar.

## Setup

Clone the repository:
git clone https://github.com/bhk97/rerank-rs.git
cd rerank-rs

Ensure your project has access to the required model and tokenizer files:
models/model.onnx
json/tokenizer.json

Download from HuggingFace: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/main/onnx

## Benchmark

Evaluated on the SciFact dev set (50 queries) using BM25 retrieval (top-50) followed by cross-encoder reranking.

| Model | MRR@50 |
|---|---|
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.845 |

Retrieval: Tantivy BM25, reranking window: top-50 candidates per query.

## Example

```rust
use rerank::{CrossEncoderReranker, Reranker, RerankerConfig};

#[tokio::main]
async fn main() -> Result<()> {
    let query = "How many people live in Berlin?";
    let docs = vec![
        "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
        "Berlin is well known for its museums.",
        "Berlin Population increased very much",
        "Berlin Population is 3M.",
    ];

    let config = RerankerConfig {
        model_path: "models/model.onnx".to_string(),
        tokenizer_path: "json/tokenizer.json".to_string(),
    };

    let reranker = CrossEncoderReranker::new(config)?;
    let ranked = reranker.rerank(query, docs, 1).await?;
    println!("{:?}", ranked);
    Ok(())
}
```

## Notes

- Requires an async runtime (Tokio).
- Paths must point to valid ONNX model and tokenizer files.
- `top_k` controls how many results are returned.
- Model and tokenizer are loaded once at initialization. First call carries the loading cost — subsequent calls do not.
- To isolate reranking latency from loading latency, run the included CLI example:
cargo run --example cli

## CLI Example
=== Reranker CLI Example ===
Commands:
query  -> set query
doc    -> add document
done   -> run reranker
exit   -> quit

1. Enter `query`, then type your query string.
2. Enter `doc` once per document to add documents individually.
3. Enter `done` to run the reranker.
4. Enter `exit` to quit.

## Output

Returns documents sorted by relevance score, highest first.

## Limitations

- Model must be in ONNX format.
- Each document is scored independently against the query; no cross-document context is used.
- Performance depends on the model. ms-marco-MiniLM-L-6-v2 is a general-purpose model not finetuned for scientific or domain-specific text.