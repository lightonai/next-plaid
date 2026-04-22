//! Browser-targeted Rust/Wasm preprocessing wrapper for the encoder worker.

use std::{cell::RefCell, collections::HashSet};

use next_plaid_preprocess::{
    build_skiplist, prepare_batch_from_tokenizer_encodings, preprocess_texts, update_token_ids,
    ColbertConfig,
};
use serde::{Deserialize, Serialize};
use tokenizers::{Encoding, Tokenizer};
use wasm_bindgen::prelude::*;

type InternalResult<T> = Result<T, String>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct PreprocessorSummary {
    query_length: usize,
    document_length: usize,
    uses_token_type_ids: bool,
    mask_token_id: u32,
    pad_token_id: u32,
    query_prefix_id: u32,
    document_prefix_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct RawPreparedInput {
    input_ids: Vec<u32>,
    attention_mask: Vec<u32>,
    token_type_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct RawPreparedDocumentInput {
    input_ids: Vec<u32>,
    attention_mask: Vec<u32>,
    token_type_ids: Option<Vec<u32>>,
    retain_row_indices: Vec<u32>,
    active_length: usize,
}

#[derive(Debug)]
struct InitializedState {
    tokenizer: Tokenizer,
    config: ColbertConfig,
    skiplist_ids: HashSet<u32>,
}

thread_local! {
    static STATE: RefCell<Option<InitializedState>> = const { RefCell::new(None) };
}

fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

fn to_js_value<T: Serialize>(value: &T) -> Result<JsValue, JsError> {
    serde_wasm_bindgen::to_value(value).map_err(|error| JsError::new(&error.to_string()))
}

fn to_js_error(error: String) -> JsError {
    JsError::new(&error)
}

fn state_not_initialized() -> String {
    "encoder preprocessor is not initialized".to_string()
}

fn with_state<T>(f: impl FnOnce(&InitializedState) -> InternalResult<T>) -> InternalResult<T> {
    STATE.with(|state| {
        let borrowed = state.borrow();
        let initialized = borrowed.as_ref().ok_or_else(state_not_initialized)?;
        f(initialized)
    })
}

fn resolve_prefix_token_id(
    tokenizer: &Tokenizer,
    configured: Option<u32>,
    prefix: &str,
    label: &str,
) -> InternalResult<u32> {
    match configured {
        Some(token_id) => Ok(token_id),
        None => tokenizer
            .token_to_id(prefix)
            .ok_or_else(|| format!("missing {label} token '{prefix}' in tokenizer")),
    }
}

fn build_summary(
    tokenizer: &Tokenizer,
    config: &ColbertConfig,
) -> InternalResult<PreprocessorSummary> {
    Ok(PreprocessorSummary {
        query_length: config.query_length,
        document_length: config.document_length,
        uses_token_type_ids: config.uses_token_type_ids,
        mask_token_id: config.mask_token_id,
        pad_token_id: config.pad_token_id,
        query_prefix_id: resolve_prefix_token_id(
            tokenizer,
            config.query_prefix_id(),
            &config.query_prefix,
            "query prefix",
        )?,
        document_prefix_id: resolve_prefix_token_id(
            tokenizer,
            config.document_prefix_id(),
            &config.document_prefix,
            "document prefix",
        )?,
    })
}

fn encode_text(state: &InitializedState, text: &str) -> InternalResult<Encoding> {
    let normalized = preprocess_texts(&state.config, &[text]);
    state
        .tokenizer
        .encode(normalized[0].clone(), true)
        .map_err(|error| error.to_string())
}

fn default_fill_values(config: &ColbertConfig, is_query: bool) -> (u32, u32) {
    if is_query && config.do_query_expansion {
        (config.mask_token_id, 1)
    } else {
        (config.pad_token_id, 0)
    }
}

fn to_u32_vec(label: &str, values: &[i64]) -> InternalResult<Vec<u32>> {
    values
        .iter()
        .map(|value| {
            u32::try_from(*value).map_err(|_| {
                format!("{label} must contain only non-negative u32 values; got {value}")
            })
        })
        .collect()
}

fn pad_row(label: &str, values: &mut Vec<u32>, target_len: usize, fill: u32) -> InternalResult<()> {
    if values.len() > target_len {
        return Err(format!(
            "{label} row length {} exceeds configured target length {target_len}",
            values.len()
        ));
    }
    values.resize(target_len, fill);
    Ok(())
}

fn prepare_raw_row(
    state: &InitializedState,
    text: &str,
    is_query: bool,
    filter_skiplist: bool,
) -> InternalResult<(RawPreparedInput, Vec<u32>, usize)> {
    let encoding = encode_text(state, text)?;
    let prepared = prepare_batch_from_tokenizer_encodings(
        &state.tokenizer,
        &state.config,
        vec![encoding],
        is_query,
        filter_skiplist,
    )
    .map_err(|error| error.to_string())?;
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
        return Err(format!("expected one prepared row, received {batch_size}"));
    }

    let target_len = if is_query {
        state.config.query_length
    } else {
        state.config.document_length
    };
    let (default_input_id, default_attention_mask) = default_fill_values(&state.config, is_query);

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
        .ok_or_else(|| "prepared batch did not include token ids".to_string())?;
    let active_length = original_lengths
        .into_iter()
        .next()
        .ok_or_else(|| "prepared batch did not include original lengths".to_string())?;

    Ok((
        RawPreparedInput {
            input_ids,
            attention_mask,
            token_type_ids,
        },
        token_ids,
        active_length,
    ))
}

fn retain_row_indices(
    state: &InitializedState,
    token_ids: &[u32],
    active_length: usize,
) -> InternalResult<Vec<u32>> {
    if active_length > token_ids.len() {
        return Err(format!(
            "active_length {active_length} exceeds token_id length {}",
            token_ids.len()
        ));
    }

    Ok(token_ids
        .iter()
        .enumerate()
        .take(active_length)
        .filter_map(|(row_index, token_id)| {
            (!state.skiplist_ids.contains(token_id)).then_some(row_index as u32)
        })
        .collect())
}

/// Initialize the shared tokenizer/config state for later query/document preparation.
#[wasm_bindgen]
pub fn init(
    tokenizer_json_bytes: Vec<u8>,
    onnx_config_json_bytes: Vec<u8>,
) -> Result<JsValue, JsError> {
    to_js_value(&init_internal(tokenizer_json_bytes, onnx_config_json_bytes).map_err(to_js_error)?)
}

fn init_internal(
    tokenizer_json_bytes: Vec<u8>,
    onnx_config_json_bytes: Vec<u8>,
) -> InternalResult<PreprocessorSummary> {
    init_panic_hook();

    let tokenizer =
        Tokenizer::from_bytes(tokenizer_json_bytes).map_err(|error| error.to_string())?;
    let mut config = ColbertConfig::from_json_bytes(&onnx_config_json_bytes)
        .map_err(|error| error.to_string())?;
    update_token_ids(&mut config, &tokenizer);
    let skiplist_ids = build_skiplist(&config, &tokenizer);
    let summary = build_summary(&tokenizer, &config)?;

    STATE.with(|state| {
        *state.borrow_mut() = Some(InitializedState {
            tokenizer,
            config,
            skiplist_ids,
        });
    });

    Ok(summary)
}

/// Prepare one query into fixed-length ONNX-ready input buffers.
#[wasm_bindgen]
pub fn prepare_query(text: &str) -> Result<JsValue, JsError> {
    to_js_value(&prepare_query_internal(text).map_err(to_js_error)?)
}

fn prepare_query_internal(text: &str) -> InternalResult<RawPreparedInput> {
    init_panic_hook();
    with_state(|state| {
        let (prepared, _token_ids, _active_length) = prepare_raw_row(state, text, true, false)?;
        Ok(prepared)
    })
}

/// Prepare one document into fixed-length ONNX-ready buffers plus retain-row metadata.
#[wasm_bindgen]
pub fn prepare_document(text: &str) -> Result<JsValue, JsError> {
    to_js_value(&prepare_document_internal(text).map_err(to_js_error)?)
}

fn prepare_document_internal(text: &str) -> InternalResult<RawPreparedDocumentInput> {
    init_panic_hook();
    with_state(|state| {
        let (prepared, token_ids, active_length) = prepare_raw_row(state, text, false, true)?;
        let retain_row_indices = retain_row_indices(state, &token_ids, active_length)?;
        Ok(RawPreparedDocumentInput {
            input_ids: prepared.input_ids,
            attention_mask: prepared.attention_mask,
            token_type_ids: prepared.token_type_ids,
            retain_row_indices,
            active_length,
        })
    })
}

/// Reset the initialized tokenizer/config state.
#[wasm_bindgen]
pub fn reset() {
    init_panic_hook();
    STATE.with(|state| {
        *state.borrow_mut() = None;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokenizers::{
        models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
        processors::bert::BertProcessing, Tokenizer,
    };

    fn fixture_tokenizer_json() -> Vec<u8> {
        let vocab = [
            ("[PAD]".to_string(), 0),
            ("[UNK]".to_string(), 1),
            ("[CLS]".to_string(), 2),
            ("[SEP]".to_string(), 3),
            ("[MASK]".to_string(), 4),
            ("[unused0]".to_string(), 5),
            ("[unused1]".to_string(), 6),
            ("alpha".to_string(), 7),
            ("beta".to_string(), 8),
            ("gamma".to_string(), 9),
            ("skip".to_string(), 10),
        ];

        let mut tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab.into())
                .unk_token("[UNK]".to_string())
                .build()
                .expect("fixture tokenizer should build"),
        );
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer.with_post_processor(Some(BertProcessing::new(
            ("[SEP]".to_string(), 3),
            ("[CLS]".to_string(), 2),
        )));

        serde_json::to_vec(&tokenizer).expect("fixture tokenizer should serialize")
    }

    fn fixture_config_json(overrides: serde_json::Value) -> Vec<u8> {
        let mut document = json!({
            "query_prefix": "[unused0]",
            "document_prefix": "[unused1]",
            "query_length": 6,
            "document_length": 6,
            "do_query_expansion": false,
            "embedding_dim": 4,
            "uses_token_type_ids": false,
            "mask_token_id": 103,
            "pad_token_id": 0,
            "skiplist_words": [],
            "do_lower_case": true
        });

        if let Some(overrides) = overrides.as_object() {
            let document_object = document
                .as_object_mut()
                .expect("fixture config must stay a JSON object");
            for (key, value) in overrides {
                document_object.insert(key.clone(), value.clone());
            }
        }

        serde_json::to_vec(&document).expect("fixture config should serialize")
    }

    fn init_fixture(config_overrides: serde_json::Value) -> PreprocessorSummary {
        init_internal(
            fixture_tokenizer_json(),
            fixture_config_json(config_overrides),
        )
        .expect("init should succeed")
    }

    #[test]
    fn init_and_reset_lifecycle_is_enforced() {
        reset();
        let summary = init_fixture(json!({}));
        assert_eq!(summary.query_prefix_id, 5);
        reset();
        let error =
            prepare_query_internal("alpha").expect_err("query prepare should fail after reset");
        assert!(error.contains("not initialized"));
    }

    #[test]
    fn query_prefix_is_inserted_after_cls() {
        reset();
        init_fixture(json!({}));

        let prepared = prepare_query_internal("alpha beta").expect("query should prepare");
        assert_eq!(prepared.input_ids, vec![2, 5, 7, 8, 3, 0]);
        assert_eq!(prepared.attention_mask, vec![1, 1, 1, 1, 1, 0]);
        assert_eq!(prepared.token_type_ids, None);
    }

    #[test]
    fn truncation_reserves_the_prefix_slot() {
        reset();
        init_fixture(json!({
            "query_length": 4,
        }));

        let prepared = prepare_query_internal("alpha beta gamma").expect("query should prepare");
        assert_eq!(prepared.input_ids, vec![2, 5, 7, 3]);
        assert_eq!(prepared.attention_mask, vec![1, 1, 1, 1]);
    }

    #[test]
    fn token_type_ids_follow_the_config() {
        reset();
        init_fixture(json!({
            "uses_token_type_ids": true,
        }));

        let prepared = prepare_query_internal("alpha").expect("query should prepare");
        assert_eq!(prepared.token_type_ids, Some(vec![0, 0, 0, 0, 0, 0]),);
    }

    #[test]
    fn query_outputs_are_fixed_length() {
        reset();
        init_fixture(json!({
            "query_length": 8,
        }));

        let prepared = prepare_query_internal("alpha").expect("query should prepare");
        assert_eq!(prepared.input_ids.len(), 8);
        assert_eq!(prepared.attention_mask.len(), 8);
        assert_eq!(prepared.input_ids[..4], [2, 5, 7, 3]);
    }

    #[test]
    fn document_retain_rows_match_skiplist_filtering() {
        reset();
        init_fixture(json!({
            "document_length": 8,
            "skiplist_words": ["skip"],
        }));

        let prepared =
            prepare_document_internal("alpha skip beta").expect("document should prepare");
        assert_eq!(prepared.input_ids, vec![2, 6, 7, 10, 8, 3, 0, 0]);
        assert_eq!(prepared.active_length, 6);
        assert_eq!(prepared.retain_row_indices, vec![0, 1, 2, 4, 5]);
    }

    #[test]
    fn invalid_tokenizer_or_config_fails_init() {
        reset();
        assert!(init_internal(b"{not-json".to_vec(), fixture_config_json(json!({}))).is_err());
        assert!(init_internal(
            fixture_tokenizer_json(),
            fixture_config_json(json!({
                "query_prefix": "[missing-prefix]",
            })),
        )
        .is_err());
    }

    #[cfg(target_arch = "wasm32")]
    mod wasm {
        use super::*;
        use wasm_bindgen_test::*;

        wasm_bindgen_test_configure!(run_in_browser);

        #[wasm_bindgen_test]
        fn wasm_init_and_query_prepare_succeeds() {
            reset();
            let summary = init_fixture(json!({}));
            assert_eq!(summary.document_prefix_id, 6);

            let prepared: RawPreparedInput = serde_wasm_bindgen::from_value(
                prepare_query("alpha beta").expect("query should prepare"),
            )
            .expect("query payload should decode");
            assert_eq!(prepared.input_ids, vec![2, 5, 7, 8, 3, 0]);
        }

        #[wasm_bindgen_test]
        fn wasm_document_prepare_returns_retain_rows() {
            reset();
            init_fixture(json!({
                "document_length": 8,
                "skiplist_words": ["skip"],
            }));

            let prepared: RawPreparedDocumentInput = serde_wasm_bindgen::from_value(
                prepare_document("alpha skip beta").expect("document should prepare"),
            )
            .expect("document payload should decode");
            assert_eq!(prepared.retain_row_indices, vec![0, 1, 2, 4, 5]);
        }

        #[wasm_bindgen_test]
        fn wasm_reset_clears_state() {
            reset();
            init_fixture(json!({}));
            reset();
            assert!(prepare_document("alpha").is_err());
        }
    }
}
