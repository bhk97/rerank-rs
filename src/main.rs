use std::time::Instant;

use anyhow::Result;
use rerank::{CrossEncoderReranker, Reranker, RerankerConfig};
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    let start_time = Instant::now();
    let query = "How many people live in Berlin?";
    let docs = vec![
        "Berlin is well known for its museums.",
        "A large metropolitan city had a population of 4123456 registered residents 
       ",
        "Berlin Population increased vey much",
        "Berlin Population is 3M.",
    ];
    let config = RerankerConfig {
        model_path: "models/model.onnx".to_string(),
        tokenizer_path: "json/tokenizer.json".to_string(),
    };
    let reranker = CrossEncoderReranker::new(config).unwrap();

    let doc_sorted = reranker.rerank(query, docs, 1).await?;
    println!("docs: {:?}", doc_sorted);
    let duration = start_time.elapsed();
    println!("Time taken: {:?}", duration);
    Ok(())
}

//  let reranker = CrossEncoderReranker::new(config).unwrap();
//     let mut query = String::new();
//     let mut docs: Vec<String> = Vec::new();

//     loop {
//         let mut input = String::new();
//         io::stdin().read_line(&mut input).unwrap();
//         let input = input.trim();

//         match input {
//             "query" => {
//                 query.clear();
//                 io::stdin().read_line(&mut query).unwrap();
//                 query = query.trim().to_string();
//             }

//             "doc" => {
//                 let mut doc = String::new();
//                 io::stdin().read_line(&mut doc).unwrap();
//                 docs.push(doc.trim().to_string());
//             }

//             "done" => {
//                 if query.is_empty() || docs.is_empty() {
//                     println!("Query or docs missing.");
//                     continue;
//                 }

//                 let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
//                 let start_time = Instant::now();
//                 let res = reranker.rerank(&query, doc_refs, 1).await.unwrap();
//                 println!("{:?}", res);
//                 let duration = start_time.elapsed();
//                 println!("Time taken: {:?}", duration);
//                 query.clear();
//                 docs.clear();
//             }

//             "exit" => break,

//             _ => println!("Invalid command"),
//         }
//     }
