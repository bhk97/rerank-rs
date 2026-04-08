# rerank-rs
rerank-rs lets Rust developers improve their search quality using local cross-encoder reranking without paying for an API or running a Python sidecar.

## Setup

Clone the repository:

```
git clone https://github.com/bhk97/rerank-rs.git
cd rerank-rs
```

Ensure your project has access to the required model and tokenizer files:

```
models/model.onnx
json/tokenizer.json
```

## Example

```rust
use rerank::{CrossEncoderReranker, Reranker, RerankerConfig};

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
```

## Notes

* Requires an async runtime such as Tokio.
* Paths must point to valid ONNX model and tokenizer files.
* `top_k` in `rerank` controls how many results are returned.
* Loading of model and tokenizer file will consume more time, for comparision
of latency please run the following:

```rust
cargo run --example cli

```

=== Reranker CLI Example ===
Commands:
query  -> set query
doc    -> add document
done   -> run reranker
exit   -> quit

# How to run the example

1. Set the Query:

```
query
```

2. Enter the query:

```
How many people live in Berlin?
```

3. Add the documents:

```
doc
```

4. Enter the documents:

```
Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.
Berlin is well known for its museums.
Berlin Population increased vey much
Berlin Population is 3M.
```

**Note All Documents cannot be added together, you need to run the doc for each individual document**
5. Run the reranker:

```
done
```

4. Exit:

```
exit
```

## Output

Returns documents sorted by relevance to the query.
