use rerank::{CrossEncoderReranker, Reranker, RerankerConfig};

#[tokio::test]
async fn test_long_document_handling() {
    let reranker = CrossEncoderReranker::new(RerankerConfig {
        model_path: "tests/fixture/models/model.onnx".into(),
        tokenizer_path: "tests/fixture/json/tokenizer.json".into(),
    })
    .unwrap();

    let query = "Rust Programming lang.";
    let long_doc = "rust programming".repeat(200);
    let docs = vec![long_doc.as_str(), "short doc"];

    let results = reranker.rerank(query, docs, Some(2), false).await.unwrap();
    // must return results without panic
    assert_eq!(results.len(), 2);
    // scores must be valid floats
    assert!(results[0].score.is_finite());
    assert!(results[1].score.is_finite());
}
