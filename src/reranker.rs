use crate::errors::RerankerError;
use crate::model;
use crate::tokenizer;
use crate::traits::RerankerConfig;
use crate::traits::{CrossEncoderReranker, RankedDocument, Reranker};
use anyhow::Result;
use ort::session::Session;
use ort::value::{TensorValueType, Value};
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
    let s = Instant::now();
    let mut model = reranker.model.lock().unwrap();
    let mut combined_scores: Vec<f32> = vec![];

    let mut windows_pairs = vec![];
    for bucket in tokenised_values.res {
        let [input_ids, attention_mask, token_type_ids] = &bucket.tensor_values[..] else {
            print!("Error in Value Extracting");
            return Err(RerankerError::InvalidInput);
        };

        println!("INP IDS GENERATED");

        windows_pairs.extend(bucket.windowed_pairs);
        let scores = return_scores(&mut model, input_ids, attention_mask, token_type_ids)?;
        println!("SCORE: {:?}", scores);
        combined_scores.extend(scores);
    }
    let e = s.elapsed();
    println!("Model inference time: {:?}", e);

    let mut aggregated_score: Vec<f32> = vec![f32::MIN; docs.len()];
    for (idx, window) in windows_pairs.iter().enumerate() {
        if aggregated_score[window.doc_index] == f32::MIN {
            aggregated_score[window.doc_index] = combined_scores[idx];
        } else {
            aggregated_score[window.doc_index] =
                aggregated_score[window.doc_index].max(combined_scores[idx]);
        }
    }
    let mut res: Vec<RankedDocument> = vec![];
    for (index, (doc, score)) in docs.iter().zip(aggregated_score.iter()).enumerate() {
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

pub fn return_scores(
    model: &mut Session,
    input_ids: &Value<TensorValueType<i64>>,
    attention_mask: &Value<TensorValueType<i64>>,
    token_type_ids: &Value<TensorValueType<i64>>,
) -> Result<Vec<f32>, RerankerError> {
    let outputs = model.run(ort::inputs![
        "input_ids" => input_ids,
        "attention_mask" => attention_mask,
        "token_type_ids" => token_type_ids
    ])?;
    let logits = outputs["logits"].try_extract_array::<f32>()?;
    let scores: Vec<f32> = logits.iter().cloned().collect();
    Ok(scores)
}
