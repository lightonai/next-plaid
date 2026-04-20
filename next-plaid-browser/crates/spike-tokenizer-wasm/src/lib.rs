//! Minimal browser-wasm spike for testing tokenizer viability.

use std::cell::RefCell;
use serde::Deserialize;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, Deserialize)]
struct SpikeConfig {
    #[serde(default)]
    do_lower_case: bool,
}

#[derive(Debug)]
struct InitializedState {
    tokenizer: Tokenizer,
    config: SpikeConfig,
}

thread_local! {
    static STATE: RefCell<Option<InitializedState>> = const { RefCell::new(None) };
}

fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

fn state_not_initialized() -> JsError {
    JsError::new("tokenizer spike is not initialized")
}

fn with_state<T>(
    f: impl FnOnce(&InitializedState) -> Result<T, JsError>,
) -> Result<T, JsError> {
    STATE.with(|state| {
        let borrowed = state.borrow();
        let initialized = borrowed.as_ref().ok_or_else(state_not_initialized)?;
        f(initialized)
    })
}

fn normalize_text(config: &SpikeConfig, text: &str) -> String {
    let trimmed = text.trim();
    if config.do_lower_case {
        trimmed.to_lowercase()
    } else {
        trimmed.to_owned()
    }
}

/// Initializes the tokenizer spike from tokenizer JSON bytes and config JSON bytes.
#[wasm_bindgen]
pub fn init(tokenizer_json: Vec<u8>, config_json: Vec<u8>) -> Result<(), JsError> {
    init_panic_hook();
    let tokenizer =
        Tokenizer::from_bytes(tokenizer_json).map_err(|error| JsError::new(&error.to_string()))?;
    let config: SpikeConfig =
        serde_json::from_slice(&config_json).map_err(|error| JsError::new(&error.to_string()))?;

    STATE.with(|state| {
        *state.borrow_mut() = Some(InitializedState { tokenizer, config });
    });

    Ok(())
}

/// Clears the initialized tokenizer/config state.
#[wasm_bindgen]
pub fn reset() {
    init_panic_hook();
    STATE.with(|state| {
        *state.borrow_mut() = None;
    });
}

/// Tokenizes one input string and returns the resulting token ids.
#[wasm_bindgen]
pub fn tokenize(text: &str) -> Result<Box<[u32]>, JsError> {
    init_panic_hook();
    with_state(|state| {
        let normalized = normalize_text(&state.config, text);
        let encoding = state
            .tokenizer
            .encode(normalized, true)
            .map_err(|error| JsError::new(&error.to_string()))?;
        Ok(encoding.get_ids().to_vec().into_boxed_slice())
    })
}

#[cfg(test)]
fn build_fixture_tokenizer_json() -> String {
    use tokenizers::models::wordpiece::WordPiece;
    use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
    use tokenizers::processors::bert::BertProcessing;

    let vocab = [
        ("[UNK]".to_string(), 0),
        ("[CLS]".to_string(), 1),
        ("[SEP]".to_string(), 2),
        ("[PAD]".to_string(), 3),
        ("[MASK]".to_string(), 4),
        ("hello".to_string(), 5),
        ("browser".to_string(), 6),
        ("rust".to_string(), 7),
        ("token".to_string(), 8),
        ("##izer".to_string(), 9),
        ("##s".to_string(), 10),
    ];

    let mut tokenizer = Tokenizer::new(
        WordPiece::builder()
            .vocab(vocab.into())
            .unk_token("[UNK]".to_string())
            .build()
            .expect("fixture tokenizer should build"),
    );
    tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
    tokenizer.with_post_processor(Some(BertProcessing::new(
        ("[SEP]".to_string(), 2),
        ("[CLS]".to_string(), 1),
    )));
    serde_json::to_string(&tokenizer).expect("fixture tokenizer should serialize")
}

#[cfg(test)]
fn fixture_config_json(do_lower_case: bool) -> Vec<u8> {
    serde_json::json!({
        "do_lower_case": do_lower_case,
    })
    .to_string()
    .into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_round_trip_tokenizes() {
        reset();
        init(
            build_fixture_tokenizer_json().into_bytes(),
            fixture_config_json(true),
        )
        .expect("init should succeed");

        let token_ids = tokenize("  Hello tokenizers  ").expect("tokenize should succeed");
        assert_eq!(&*token_ids, &[1, 5, 8, 9, 10, 2]);
    }

    #[cfg(target_arch = "wasm32")]
    mod wasm {
        use super::*;
        use wasm_bindgen_test::*;

        wasm_bindgen_test_configure!(run_in_browser);

        #[wasm_bindgen_test]
        fn wasm_round_trip_tokenizes() {
            reset();
            init(
                build_fixture_tokenizer_json().into_bytes(),
                fixture_config_json(true),
            )
            .expect("init should succeed");

            let token_ids = tokenize("  Hello tokenizers  ").expect("tokenize should succeed");
            assert_eq!(&*token_ids, &[1, 5, 8, 9, 10, 2]);
        }

        #[wasm_bindgen_test]
        fn wasm_unknown_token_falls_back_to_unk() {
            reset();
            init(
                build_fixture_tokenizer_json().into_bytes(),
                fixture_config_json(false),
            )
            .expect("init should succeed");

            let token_ids = tokenize("mystery").expect("tokenize should succeed");
            assert_eq!(&*token_ids, &[1, 0, 2]);
        }
    }
}
