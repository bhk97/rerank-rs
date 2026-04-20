use crate::errors::RerankerError;
use crate::model;
use crate::tokenizer;
use crate::traits::RerankerConfig;
use crate::traits::{CrossEncoderReranker, RankedDocument, Reranker};
use anyhow::Result;
use ndarray::Array1;
use ort::session::Session;
use ort::value::{TensorValueType, Value};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::usize;
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
    /// Rerank the documents based on the query
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<&str>,
        top_n: Option<usize>,
        normalize_scores: bool,
    ) -> Result<Vec<RankedDocument>, RerankerError> {
        rerank_logic(self, query, documents, top_n, normalize_scores)
    }
}

pub(crate) fn rerank_logic(
    reranker: &CrossEncoderReranker,
    query: &str,
    docs: Vec<&str>,
    top_n: Option<usize>,
    normalize_scores: bool,
) -> Result<Vec<RankedDocument>, RerankerError> {
    let tokenised_values = tokenizer::tokenise_data(query, docs.clone(), &reranker.tokenizer)
        .map_err(|_| RerankerError::FileLoad)?;
    let s = Instant::now();
    let mut model = reranker
        .model
        .lock()
        .map_err(|_| RerankerError::ModelLoad)?;
    let mut combined_scores: Vec<f32> = vec![];

    let mut windows_pairs = vec![];
    for bucket in tokenised_values.res {
        let [input_ids, attention_mask, token_type_ids] = &bucket.tensor_values[..] else {
            print!("Error in Value Extracting");
            return Err(RerankerError::InvalidInput);
        };

        println!("INP IDS GENERATED");

        windows_pairs.extend(bucket.windowed_pairs);
        let scores = return_scores(
            &mut model,
            input_ids,
            attention_mask,
            token_type_ids,
            normalize_scores,
        )?;
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
    match top_n {
        Some(top_n) => {
            res.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        None => {}
    }

    match top_n {
        Some(top_n) => res.truncate(top_n),
        None => {}
    }
    Ok(res)
}

pub fn return_scores(
    model: &mut Session,
    input_ids: &Value<TensorValueType<i64>>,
    attention_mask: &Value<TensorValueType<i64>>,
    token_type_ids: &Value<TensorValueType<i64>>,
    normalize_scores: bool,
) -> Result<Vec<f32>, RerankerError> {
    let outputs = model.run(ort::inputs![
        "input_ids" => input_ids,
        "attention_mask" => attention_mask,
        "token_type_ids" => token_type_ids
    ])?;
    let logits = outputs["logits"].try_extract_array::<f32>()?;
    let scores: Vec<f32> = logits.iter().cloned().collect();
    if normalize_scores {
        let sigmoids = sigmoid_conversion(scores);
        Ok(sigmoids)
    } else {
        Ok(scores)
    }
}

fn sigmoid_conversion(z: Vec<f32>) -> Vec<f32> {
    z.into_iter()
        .map(|x| {
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            }
        })
        .collect()
}
