use std::{clone, time::Instant};

use crate::{
    errors::RerankerError,
    traits::{BucketResult, TokenizerResult, WindowedPair},
};
use ndarray::Array2;
use ort::value::{TensorValueType, Value};
use tokenizers::Tokenizer;

const CLS_TOKEN: i64 = 101;
const SEP_TOKEN: i64 = 102;
const PAD_TOKEN: i64 = 0;
const MAX_SEQ_LEN: usize = 512;

pub fn tokenise_data(
    query: &str,
    docs: Vec<&str>,
    tokenizer: &Tokenizer,
) -> Result<BucketResult, RerankerError> {
    let query_encoding = tokenizer
        .encode(query, false)
        .map_err(RerankerError::Tokenizer)?;
    let query_ids: Vec<i64> = query_encoding.get_ids().iter().map(|&x| x as i64).collect();
    let max_doc_tokens = MAX_SEQ_LEN - 3 - query_ids.len();

    let mut all_windows: Vec<WindowedPair> = Vec::new();

    for (doc_idx, doc) in docs.iter().enumerate() {
        let doc_encoding = tokenizer
            .encode(*doc, false)
            .map_err(RerankerError::Tokenizer)?;
        let doc_ids: Vec<i64> = doc_encoding.get_ids().iter().map(|&x| x as i64).collect();

        let windows = chunk_doc_ids(&doc_ids, max_doc_tokens, 64);

        for (win_idx, window_doc_ids) in windows.iter().enumerate() {
            let (token_ids, attention_mask, type_ids) = build_tensors(&query_ids, window_doc_ids);

            let window_token_ids_u32: Vec<u32> = window_doc_ids.iter().map(|&x| x as u32).collect();
            let window_text = tokenizer
                .decode(&window_token_ids_u32, true)
                .unwrap_or_else(|_| doc.to_string());

            all_windows.push(WindowedPair {
                query_doc_pair: (query.to_string(), window_text),
                doc_index: doc_idx,
                window_index: win_idx,
                token_ids,
                attention_mask,
                type_ids,
            });
        }
    }

    all_windows.sort_by(|a, b| a.token_ids.len().cmp(&b.token_ids.len()));
    let buckets = build_buckets(all_windows);
    // let ans = build_inputs(buckets[0].clone());
    // ans
    let mut tensor_values: Vec<TokenizerResult> = Vec::new();
    for bucket in buckets.iter() {
        if bucket.len() == 0 {
            continue;
        }
        tensor_values.push(build_inputs(bucket.clone()).unwrap());
    }
    Ok(BucketResult { res: tensor_values })
}

fn chunk_doc_ids(doc_ids: &[i64], window_size: usize, overlap: usize) -> Vec<Vec<i64>> {
    if doc_ids.len() <= window_size {
        return vec![doc_ids.to_vec()];
    }

    let mut windows: Vec<Vec<i64>> = Vec::new();
    let step = window_size - overlap;
    let mut start = 0;

    while start < doc_ids.len() {
        let end = (start + window_size).min(doc_ids.len());
        windows.push(doc_ids[start..end].to_vec());
        if end == doc_ids.len() {
            break;
        }
        start += step;
    }

    windows
}

fn build_tensors(query_ids: &[i64], doc_window_ids: &[i64]) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let total_len = 1 + query_ids.len() + 1 + doc_window_ids.len() + 1;

    let mut token_ids = Vec::with_capacity(total_len);
    let mut type_ids = Vec::with_capacity(total_len);

    // [CLS]
    token_ids.push(CLS_TOKEN);
    type_ids.push(0i64);

    // query tokens — type 0
    for &id in query_ids {
        token_ids.push(id);
        type_ids.push(0i64);
    }

    // [SEP] after query — type 0
    token_ids.push(SEP_TOKEN);
    type_ids.push(0i64);

    // doc window tokens — type 1
    for &id in doc_window_ids {
        token_ids.push(id);
        type_ids.push(1i64);
    }

    // [SEP] after doc — type 1
    token_ids.push(SEP_TOKEN);
    type_ids.push(1i64);

    // attention mask — 1 for every real token, no padding here
    let attention_mask = vec![1i64; token_ids.len()];

    (token_ids, attention_mask, type_ids)
}

fn build_inputs(all_windows: Vec<WindowedPair>) -> Result<TokenizerResult, RerankerError> {
    let max_len = all_windows
        .iter()
        .map(|w| w.token_ids.len())
        .max()
        .unwrap_or(0);

    let max_len = max_len.min(MAX_SEQ_LEN);
    println!("max_len: {}", max_len);

    let batch_size = all_windows.len();

    let mut flat_ids: Vec<i64> = Vec::with_capacity(batch_size * max_len);
    let mut flat_mask: Vec<i64> = Vec::with_capacity(batch_size * max_len);
    let mut flat_type_ids: Vec<i64> = Vec::with_capacity(batch_size * max_len);

    for window in &all_windows {
        let pad_len = max_len - window.token_ids.len();

        flat_ids.extend_from_slice(&window.token_ids);
        flat_ids.extend(std::iter::repeat(PAD_TOKEN).take(pad_len));

        flat_mask.extend_from_slice(&window.attention_mask);
        flat_mask.extend(std::iter::repeat(0i64).take(pad_len));

        flat_type_ids.extend_from_slice(&window.type_ids);
        flat_type_ids.extend(std::iter::repeat(0i64).take(pad_len));
    }

    let input_ids = Array2::from_shape_vec((batch_size, max_len), flat_ids)
        .map_err(|_| RerankerError::ShapeError)?;
    let attention_mask = Array2::from_shape_vec((batch_size, max_len), flat_mask)
        .map_err(|_| RerankerError::ShapeError)?;
    let token_type_ids = Array2::from_shape_vec((batch_size, max_len), flat_type_ids)
        .map_err(|_| RerankerError::ShapeError)?;

    let input_ids = Value::from_array(input_ids)?;
    let attention_mask = Value::from_array(attention_mask)?;
    let token_type_ids = Value::from_array(token_type_ids)?;

    Ok(TokenizerResult {
        tensor_values: vec![input_ids, attention_mask, token_type_ids],
        windowed_pairs: all_windows,
    })
}

fn build_buckets(all_windows: Vec<WindowedPair>) -> Vec<Vec<WindowedPair>> {
    let mut buckets: Vec<Vec<WindowedPair>> = vec![vec![]; 5];

    for win in &all_windows {
        match win.token_ids.len() {
            len if len <= 100 => buckets[0].push(win.clone()),
            len if len <= 200 => buckets[1].push(win.clone()),
            len if len <= 300 => buckets[2].push(win.clone()),
            len if len <= 400 => buckets[3].push(win.clone()),
            len if len > 400 => buckets[4].push(win.clone()),
            _ => unreachable!(),
        }
    }
    buckets
}
