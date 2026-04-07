use crate::errors::RerankerError;
use crate::model;
use crate::tokenizer;
use crate::traits::RerankerConfig;
use crate::traits::{CrossEncoderReranker, ModelAndTokenizer, RankedDocument, Reranker};
use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokenizers::Tokenizer;
impl CrossEncoderReranker {
    pub fn new(config: RerankerConfig) -> Result<Self, RerankerError> {
        let model = Arc::new(Mutex::new(model::load_model(&config.model_path)?));
        let tokenizer =
            Tokenizer::from_file(&config.tokenizer_path).map_err(RerankerError::Tokenizer)?;

        Ok(Self { model, tokenizer })
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
    let tokenised_values = tokenizer::tokenise_data(query, docs.clone(), &reranker.tokenizer)
        .map_err(|_| RerankerError::FileLoad)?;
    let [input_ids, attention_mask, token_type_ids] = &tokenised_values[..] else {
        return Err(RerankerError::InvalidInput);
    };
    let mut model = reranker.model.lock().unwrap();

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
