use crate::errors::RerankerError;
use crate::model;
use crate::tokenizer;
use crate::traits::RerankerConfig;
use crate::traits::{CrossEncoderReranker, RankedDocument, Reranker};
use anyhow::Result;
impl CrossEncoderReranker {
    pub fn new(config: RerankerConfig) -> Self {
        Self { config }
    }
}
impl Reranker for CrossEncoderReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<&str>,
        top_n: usize,
    ) -> Result<Vec<RankedDocument>, RerankerError> {
        rerank_logic(self, query, documents, top_n)
    }
}

pub fn rerank_logic(
    reranker: &CrossEncoderReranker,
    query: &str,
    docs: Vec<&str>,
    top_n: usize,
) -> Result<Vec<RankedDocument>, RerankerError> {
    let model_path = &reranker.config.model_path;
    let json_path = &reranker.config.tokenizer_path;
    let mut model = model::load_model(model_path)?;
    let tokenised_values = tokenizer::tokenise_data(query, docs.clone(), json_path)
        .map_err(|_| RerankerError::FileLoad)?;

    let [input_ids, attention_mask, token_type_ids] = &tokenised_values[..] else {
        return Err(RerankerError::InvalidInput);
    };

    let outputs = model.run(ort::inputs![
        "input_ids" => input_ids,
        "attention_mask" => attention_mask,
        "token_type_ids" => token_type_ids
    ])?;

    let logits = outputs["logits"].try_extract_array::<f32>()?;
    let scores: Vec<f32> = logits.iter().cloned().collect();
    let mut res: Vec<RankedDocument> = vec![];
    for (index, (doc, score)) in docs.iter().zip(scores.iter()).enumerate() {
        res.push(RankedDocument {
            index,
            text: doc.to_string(),
            score: *score,
        });
    }

    res.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    res.truncate(top_n);
    Ok(res)
}
