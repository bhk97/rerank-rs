use rerank::{CrossEncoderReranker, Reranker, RerankerConfig};
use tokio::test;
#[tokio::test]
async fn test_top_n() {
    let reranker = CrossEncoderReranker::new(RerankerConfig {
        model_path: "tests/fixture/models/model.onnx".into(),
        tokenizer_path: "tests/fixture/json/tokenizer.json".into(),
    })
    .unwrap();

    let docs = vec!["doc one", "doc two", "doc three", "doc four", "doc five"];
    let results = reranker.rerank("query", docs, 3).await.unwrap();

    assert_eq!(results.len(), 3);
}
