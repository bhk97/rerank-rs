use std::{io, time::Instant};

use rerank::{CrossEncoderReranker, Reranker, RerankerConfig};

#[tokio::main]
async fn main() {
    println!("=== Reranker CLI Example ===");
    println!("Commands:");
    println!("query  -> set query");
    println!("doc    -> add document");
    println!("done   -> run reranker");
    println!("exit   -> quit\n");

    let config = RerankerConfig {
        model_path: "models/model.onnx".to_string(),
        tokenizer_path: "json/tokenizer.json".to_string(),
    };

    let reranker = CrossEncoderReranker::new(config).expect("Failed to initialize reranker");

    let mut query = String::new();
    let mut docs: Vec<String> = Vec::new();

    loop {
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        match input {
            "query" => {
                println!("Enter query:");
                query.clear();

                io::stdin().read_line(&mut query).unwrap();
                query = query.trim().to_string();

                println!("Query set.\n");
            }

            "doc" => {
                println!("Enter document:");

                let mut doc = String::new();
                io::stdin().read_line(&mut doc).unwrap();

                docs.push(doc.trim().to_string());

                println!("Doc added. Total docs: {}\n", docs.len());
            }

            "done" => {
                if query.is_empty() || docs.is_empty() {
                    println!("Query or docs missing.\n");
                    continue;
                }

                println!("Running reranker...\n");

                let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();

                let start_time = Instant::now();

                let res = reranker
                    .rerank(&query, doc_refs, 1)
                    .await
                    .expect("Rerank failed");

                println!("Results:\n{:#?}", res);
                println!("Time taken: {:?}\n", start_time.elapsed());

                // Reset state after run
                query.clear();
                docs.clear();
            }

            "exit" => {
                println!("Exiting...");
                break;
            }

            _ => {
                println!("Invalid command.\n");
            }
        }
    }
}
