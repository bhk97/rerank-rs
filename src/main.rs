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
         Reranking is a critical stage in contemporary information retrieval (IR) systems, 
         improving the relevance of the user-presented final results by honing initial candidate sets. 
         This paper is a thorough guide to examine the changing reranker landscape and offer a clear 
         view of the advancements made in reranking methods. We present a comprehensive survey of 
         reranking models employed in IR, particularly within modern Retrieval Augmented Generation 
         (RAG) pipelines, where retrieved documents notably influence output quality.
         We embark on a chronological journey through the historical trajectory of reranking techniques,
          starting with foundational approaches, before exploring the wide range of sophisticated neural 
          network architectures such as cross-encoders, sequence-generation models like T5, and Graph Neural
           Networks (GNNs) utilized for structural information. Recognizing the computational cost of 
           advancing neural rerankers, we analyze techniques for enhancing efficiency, notably knowledge 
           distillation for creating competitive, lighter alternatives. Furthermore, we map the emerging 
           territory of integrating Large Language Models (LLMs) in reranking, examining novel prompting
            strategies and fine-tuning tactics. This survey seeks to elucidate the fundamental ideas, 
            relative effectiveness, computational features, and real-world trade-offs of various reranking 
            strategies. The survey provides a structured synthesis of the diverse reranking paradigms, 
            highlighting their underlying 
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
