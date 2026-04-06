use anyhow::Result;
mod model;
mod reranker;
mod tokenizer;

fn main() -> Result<()> {
    let query = "How many people live in Berlin?";
    let docs = vec![
        "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
        "Berlin is well known for its museums.",
    ];
    let doc_sorted = reranker::rerank(query, docs)?;
    println!("Sorted docs: {:?}", doc_sorted);

    Ok(())
}
