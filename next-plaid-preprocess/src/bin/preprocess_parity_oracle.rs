use std::{collections::HashSet, env, fs, io, path::Path};

use next_plaid_preprocess::{
    build_skiplist, prepare_batch_from_tokenizer_encodings, preprocess_texts, update_token_ids,
    ColbertConfig,
};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
struct CaseInput {
    id: String,
    kind: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct OracleInput {
    cases: Vec<CaseInput>,
}

#[derive(Debug, Serialize)]
struct PreparedCase {
    id: String,
    kind: String,
    input_ids: Vec<u32>,
    attention_mask: Vec<u32>,
    token_type_ids: Option<Vec<u32>>,
    retain_row_indices: Option<Vec<u32>>,
    active_length: usize,
}

#[derive(Debug, Serialize)]
struct OracleOutput {
    cases: Vec<PreparedCase>,
}

type Error = Box<dyn std::error::Error>;
type Result<T> = std::result::Result<T, Error>;

fn invalid_data(message: impl Into<String>) -> Error {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

fn read_tokenizer(path: &Path) -> Result<Tokenizer> {
    let bytes = fs::read(path)?;
    Tokenizer::from_bytes(bytes).map_err(|error| invalid_data(error.to_string()))
}

fn read_config(path: &Path, tokenizer: &Tokenizer) -> Result<ColbertConfig> {
    let bytes = fs::read(path)?;
    let mut config = ColbertConfig::from_json_bytes(&bytes)?;
    update_token_ids(&mut config, tokenizer);
    Ok(config)
}

fn to_u32_vec(label: &str, values: &[i64]) -> Result<Vec<u32>> {
    values
        .iter()
        .map(|value| {
            u32::try_from(*value).map_err(|_| {
                invalid_data(format!(
                    "{label} must contain only non-negative u32 values; got {value}"
                ))
            })
        })
        .collect()
}

fn pad_row(label: &str, values: &mut Vec<u32>, target_len: usize, fill: u32) -> Result<()> {
    if values.len() > target_len {
        return Err(invalid_data(format!(
            "{label} row length {} exceeds configured target length {target_len}",
            values.len()
        )));
    }
    values.resize(target_len, fill);
    Ok(())
}

fn default_fill_values(config: &ColbertConfig, is_query: bool) -> (u32, u32) {
    if is_query && config.do_query_expansion {
        (config.mask_token_id, 1)
    } else {
        (config.pad_token_id, 0)
    }
}

fn retain_row_indices(
    token_ids: &[u32],
    active_length: usize,
    skiplist_ids: &HashSet<u32>,
) -> Result<Vec<u32>> {
    if active_length > token_ids.len() {
        return Err(invalid_data(format!(
            "active_length {active_length} exceeds token_id length {}",
            token_ids.len()
        )));
    }

    Ok(token_ids
        .iter()
        .enumerate()
        .take(active_length)
        .filter_map(|(row_index, token_id)| {
            (!skiplist_ids.contains(token_id)).then_some(row_index as u32)
        })
        .collect())
}

fn prepare_case(
    tokenizer: &Tokenizer,
    config: &ColbertConfig,
    skiplist_ids: &HashSet<u32>,
    case: CaseInput,
) -> Result<PreparedCase> {
    let is_query = match case.kind.as_str() {
        "query" => true,
        "document" => false,
        other => return Err(invalid_data(format!("unsupported case kind: {other}"))),
    };
    let filter_skiplist = !is_query;
    let normalized = preprocess_texts(config, &[case.text.as_str()]);
    let encoding = tokenizer
        .encode(normalized[0].clone(), true)
        .map_err(|error| invalid_data(error.to_string()))?;
    let prepared = prepare_batch_from_tokenizer_encodings(
        tokenizer,
        config,
        vec![encoding],
        is_query,
        filter_skiplist,
    )?;

    let (
        batch_size,
        batch_max_len,
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_token_ids,
        original_lengths,
        _is_query,
        _filter_skiplist,
    ) = prepared.into_parts();

    if batch_size != 1 {
        return Err(invalid_data(format!(
            "expected one prepared row, received {batch_size}"
        )));
    }

    let target_len = if is_query {
        config.query_length
    } else {
        config.document_length
    };
    let (default_input_id, default_attention_mask) = default_fill_values(config, is_query);

    let mut input_ids = to_u32_vec("input_ids", &all_input_ids[..batch_max_len])?;
    let mut attention_mask = to_u32_vec("attention_mask", &all_attention_mask[..batch_max_len])?;
    let mut token_type_ids = all_token_type_ids
        .map(|values| to_u32_vec("token_type_ids", &values[..batch_max_len]))
        .transpose()?;

    pad_row("input_ids", &mut input_ids, target_len, default_input_id)?;
    pad_row(
        "attention_mask",
        &mut attention_mask,
        target_len,
        default_attention_mask,
    )?;
    if let Some(token_type_ids) = token_type_ids.as_mut() {
        pad_row("token_type_ids", token_type_ids, target_len, 0)?;
    }

    let token_ids = all_token_ids
        .into_iter()
        .next()
        .ok_or_else(|| invalid_data("prepared batch did not include token ids"))?;
    let active_length = original_lengths
        .into_iter()
        .next()
        .ok_or_else(|| invalid_data("prepared batch did not include original lengths"))?;
    let retain_row_indices = if is_query {
        None
    } else {
        Some(retain_row_indices(&token_ids, active_length, skiplist_ids)?)
    };

    Ok(PreparedCase {
        id: case.id,
        kind: case.kind,
        input_ids,
        attention_mask,
        token_type_ids,
        retain_row_indices,
        active_length,
    })
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().collect();
    if args.len() != 4 {
        return Err(invalid_data(
            "usage: preprocess-parity-oracle <tokenizer.json> <onnx_config.json> <cases.json>",
        ));
    }

    let tokenizer = read_tokenizer(Path::new(&args[1]))?;
    let config = read_config(Path::new(&args[2]), &tokenizer)?;
    let skiplist_ids = build_skiplist(&config, &tokenizer);
    let input: OracleInput = serde_json::from_slice(&fs::read(Path::new(&args[3]))?)?;
    let cases = input
        .cases
        .into_iter()
        .map(|case| prepare_case(&tokenizer, &config, &skiplist_ids, case))
        .collect::<Result<Vec<_>>>()?;

    println!("{}", serde_json::to_string_pretty(&OracleOutput { cases })?);
    Ok(())
}
