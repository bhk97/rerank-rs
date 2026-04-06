use anyhow::{Result, anyhow};
use ndarray::Array2;
use ort::value::{TensorValueType, Value};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

pub fn tokenise_data(query: &str, docs: Vec<&str>) -> Result<Vec<Value<TensorValueType<i64>>>> {
    let mut token = Tokenizer::from_file("json/tokenizer.json").map_err(|e| anyhow!(e))?;
    let max_len = 256;

    token.with_truncation(Some(TruncationParams {
        max_length: max_len,
        ..Default::default()
    }));

    token.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(max_len),
        ..Default::default()
    }));

    let mut ids: Vec<Vec<i64>> = Vec::new();
    let mut mask: Vec<Vec<i64>> = Vec::new();
    let mut type_ids: Vec<Vec<i64>> = Vec::new();

    for d in docs {
        let encoding = token.encode((query, d), true).map_err(|e| anyhow!(e))?;
        ids.push(encoding.get_ids().iter().map(|&x| x as i64).collect());
        mask.push(
            encoding
                .get_attention_mask()
                .iter()
                .map(|&x| x as i64)
                .collect(),
        );
        type_ids.push(encoding.get_type_ids().iter().map(|&x| x as i64).collect());
    }

    let batch_size = ids.len();
    let seq_len = ids[0].len();
    let flat_ids: Vec<i64> = ids.into_iter().flatten().collect();
    let flat_mask: Vec<i64> = mask.into_iter().flatten().collect();
    let flat_type_ids: Vec<i64> = type_ids.into_iter().flatten().collect();

    let input_ids: Array2<i64> = Array2::from_shape_vec((batch_size, seq_len), flat_ids)?;
    let attention_mask: Array2<i64> = Array2::from_shape_vec((batch_size, seq_len), flat_mask)?;
    let token_type_ids: Array2<i64> = Array2::from_shape_vec((batch_size, seq_len), flat_type_ids)?;

    let input_ids = Value::from_array(input_ids)?;
    let attention_mask = Value::from_array(attention_mask)?;
    let token_type_ids = Value::from_array(token_type_ids)?;
    let res: Vec<Value<TensorValueType<i64>>> = vec![input_ids, attention_mask, token_type_ids];
    Ok(res)
}
