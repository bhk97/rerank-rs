use rerank::{CrossEncoderReranker, Reranker, RerankerConfig};
use tokio::test;

#[tokio::test]
async fn test_ranking_order() {
    let reranker = CrossEncoderReranker::new(RerankerConfig {
        model_path: "tests/fixture/models/model.onnx".into(),
        tokenizer_path: "tests/fixture/json/tokenizer.json".into(),
    })
    .unwrap();

    let query = "what is the capital of France";
    let docs = vec![
        "Paris is the capital city of France",   // clearly relevant
        "London is the capital of England",      // irrelevant
        "France is a country in Western Europe", // tangentially related
    ];

    let results = reranker.rerank(query, docs, 3).await.unwrap();
    assert_eq!(results[0].index, 0);
    assert!(results[0].score > results[1].score);
    assert!(results[1].score > results[2].score);
}
