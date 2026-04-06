use crate::model;
use crate::tokenizer;
use anyhow::Result;
use ort::value::Value;

pub fn rerank(query: &str, docs: Vec<&str>) -> Result<Vec<String>> {
    let mut model = model::load_model()?;
    let tokenised_values = tokenizer::tokenise_data(query, docs.clone())?;

    let [input_ids, attention_mask, token_type_ids] = &tokenised_values[..] else {
        panic!(
            "Expected exactly 3 values, but got {}",
            tokenised_values.len()
        )
    };

    let outputs = model.run(ort::inputs![
        "input_ids" => input_ids,
        "attention_mask" => attention_mask,
        "token_type_ids" => token_type_ids
    ])?;

    let logits = outputs["logits"].try_extract_array::<f32>()?;
    let scores: Vec<f32> = logits.iter().cloned().collect();

    let mut ranked_results: Vec<(String, f32)> = docs
        .into_iter()
        .zip(scores.into_iter())
        .map(|(doc, score)| (doc.to_string(), score))
        .collect();

    ranked_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_docs: Vec<String> = ranked_results.into_iter().map(|(doc, _)| doc).collect();

    println!("Top document: {:?}", sorted_docs.first());

    Ok(sorted_docs)
}
