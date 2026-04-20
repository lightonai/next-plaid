use std::collections::HashSet;

use anyhow::Result;
use tokenizers::{Encoding, Tokenizer};

use crate::ColbertConfig;

#[derive(Debug, Clone)]
pub struct PreparedDocumentBatch {
    pub batch_size: usize,
    pub batch_max_len: usize,
    pub all_input_ids: Vec<i64>,
    pub all_attention_mask: Vec<i64>,
    pub all_token_type_ids: Option<Vec<i64>>,
    pub all_token_ids: Vec<Vec<u32>>,
    pub original_lengths: Vec<usize>,
    pub is_query: bool,
    pub filter_skiplist: bool,
}

#[derive(Debug, Clone)]
pub struct TokenizedDocument {
    pub ids: Vec<u32>,
    pub type_ids: Vec<u32>,
}

impl PreparedDocumentBatch {
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn batch_max_len(&self) -> usize {
        self.batch_max_len
    }
}

pub fn preprocess_texts(config: &ColbertConfig, texts: &[&str]) -> Vec<String> {
    if config.do_lower_case {
        texts.iter().map(|t| t.trim().to_lowercase()).collect()
    } else {
        texts.iter().map(|t| t.trim().to_string()).collect()
    }
}

pub fn update_token_ids(config: &mut ColbertConfig, tokenizer: &Tokenizer) {
    if config.mask_token_id == 103 {
        if let Some(mask_id) = tokenizer.token_to_id("[MASK]") {
            config.mask_token_id = mask_id;
        } else if let Some(mask_id) = tokenizer.token_to_id("<mask>") {
            config.mask_token_id = mask_id;
        }
    }
    if config.pad_token_id == 0 {
        if let Some(pad_id) = tokenizer.token_to_id("[PAD]") {
            config.pad_token_id = pad_id;
        } else if let Some(pad_id) = tokenizer.token_to_id("<pad>") {
            config.pad_token_id = pad_id;
        }
    }
}

pub fn build_skiplist(config: &ColbertConfig, tokenizer: &Tokenizer) -> HashSet<u32> {
    let mut skiplist_ids = HashSet::new();
    for word in &config.skiplist_words {
        if let Some(token_id) = tokenizer.token_to_id(word) {
            skiplist_ids.insert(token_id);
        }
    }
    skiplist_ids
}

pub fn prepare_batch_from_tokenized_documents(
    tokenizer: &Tokenizer,
    config: &ColbertConfig,
    batch_docs: Vec<TokenizedDocument>,
    is_query: bool,
    filter_skiplist: bool,
) -> Result<PreparedDocumentBatch> {
    let (prefix_token_id, max_length) = resolve_prefix_token_id(tokenizer, config, is_query)?;
    let truncate_limit = max_length.saturating_sub(1);
    let mut batch_max_len = 0usize;
    for doc in &batch_docs {
        let effective_len = if doc.ids.len() > truncate_limit {
            max_length
        } else {
            doc.ids.len() + 1
        };
        batch_max_len = batch_max_len.max(effective_len);
    }
    if is_query && config.do_query_expansion {
        batch_max_len = max_length;
    }

    let batch_size = batch_docs.len();
    let (default_input_id, default_attention) = default_fill_values(config, is_query);
    let mut all_input_ids: Vec<i64> = vec![default_input_id; batch_size * batch_max_len];
    let mut all_attention_mask: Vec<i64> = vec![default_attention; batch_size * batch_max_len];
    let mut all_token_type_ids: Vec<i64> = vec![0; batch_size * batch_max_len];
    let mut all_token_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
    let mut original_lengths: Vec<usize> = Vec::with_capacity(batch_size);

    for (row_idx, doc) in batch_docs.into_iter().enumerate() {
        let row_start = row_idx * batch_max_len;
        let real_len = doc.ids.len().max(1);
        let (content_prefix_len, keep_sep) = content_prefix_plan(real_len, truncate_limit);
        let final_len = final_length(real_len, max_length, keep_sep);
        original_lengths.push(final_len);

        all_input_ids[row_start] = doc.ids[0] as i64;
        all_attention_mask[row_start] = 1;
        all_token_type_ids[row_start] = doc.type_ids[0] as i64;

        all_input_ids[row_start + 1] = prefix_token_id as i64;
        all_attention_mask[row_start + 1] = 1;
        all_token_type_ids[row_start + 1] = 0;

        let mut token_ids_vec: Vec<u32> = Vec::with_capacity(final_len);
        token_ids_vec.push(doc.ids[0]);
        token_ids_vec.push(prefix_token_id);

        let mut write_pos = row_start + 2;
        for src_idx in 1..content_prefix_len {
            all_input_ids[write_pos] = doc.ids[src_idx] as i64;
            all_attention_mask[write_pos] = 1;
            all_token_type_ids[write_pos] = doc.type_ids[src_idx] as i64;
            token_ids_vec.push(doc.ids[src_idx]);
            write_pos += 1;
        }

        if keep_sep {
            let sep_idx = real_len - 1;
            all_input_ids[write_pos] = doc.ids[sep_idx] as i64;
            all_attention_mask[write_pos] = 1;
            all_token_type_ids[write_pos] = doc.type_ids[sep_idx] as i64;
            token_ids_vec.push(doc.ids[sep_idx]);
        }

        all_token_ids.push(token_ids_vec);
    }

    Ok(PreparedDocumentBatch {
        batch_size,
        batch_max_len,
        all_input_ids,
        all_attention_mask,
        all_token_type_ids: if config.uses_token_type_ids {
            Some(all_token_type_ids)
        } else {
            None
        },
        all_token_ids,
        original_lengths,
        is_query,
        filter_skiplist,
    })
}

pub fn prepare_batch_from_tokenizer_encodings(
    tokenizer: &Tokenizer,
    config: &ColbertConfig,
    batch_encodings: Vec<Encoding>,
    is_query: bool,
    filter_skiplist: bool,
) -> Result<PreparedDocumentBatch> {
    let (prefix_token_id, max_length) = resolve_prefix_token_id(tokenizer, config, is_query)?;
    let truncate_limit = max_length.saturating_sub(1);
    let real_lengths: Vec<usize> = batch_encodings
        .iter()
        .map(|encoding| {
            encoding
                .get_attention_mask()
                .iter()
                .take_while(|&&v| v != 0)
                .count()
                .max(1)
        })
        .collect();

    let mut batch_max_len = 0usize;
    for &real_len in &real_lengths {
        let effective_len = if real_len > truncate_limit {
            max_length
        } else {
            real_len + 1
        };
        batch_max_len = batch_max_len.max(effective_len);
    }

    if is_query && config.do_query_expansion {
        batch_max_len = max_length;
    }

    let batch_size = batch_encodings.len();
    let (default_input_id, default_attention) = default_fill_values(config, is_query);
    let mut all_input_ids: Vec<i64> = vec![default_input_id; batch_size * batch_max_len];
    let mut all_attention_mask: Vec<i64> = vec![default_attention; batch_size * batch_max_len];
    let mut all_token_type_ids: Vec<i64> = vec![0; batch_size * batch_max_len];
    let mut all_token_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
    let mut original_lengths: Vec<usize> = Vec::with_capacity(batch_size);

    for (row_idx, (encoding, &real_len)) in batch_encodings.into_iter().zip(&real_lengths).enumerate() {
        let row_start = row_idx * batch_max_len;
        let ids = encoding.get_ids();
        let masks = encoding.get_attention_mask();
        let type_ids = encoding.get_type_ids();

        let (content_prefix_len, keep_sep) = content_prefix_plan(real_len, truncate_limit);
        let final_len = final_length(real_len, max_length, keep_sep);
        original_lengths.push(final_len);

        all_input_ids[row_start] = ids[0] as i64;
        all_attention_mask[row_start] = masks[0] as i64;
        all_token_type_ids[row_start] = type_ids[0] as i64;

        all_input_ids[row_start + 1] = prefix_token_id as i64;
        all_attention_mask[row_start + 1] = 1;
        all_token_type_ids[row_start + 1] = 0;

        let mut token_ids_vec: Vec<u32> = Vec::with_capacity(final_len);
        token_ids_vec.push(ids[0]);
        token_ids_vec.push(prefix_token_id);

        let mut write_pos = row_start + 2;
        for src_idx in 1..content_prefix_len {
            all_input_ids[write_pos] = ids[src_idx] as i64;
            all_attention_mask[write_pos] = masks[src_idx] as i64;
            all_token_type_ids[write_pos] = type_ids[src_idx] as i64;
            token_ids_vec.push(ids[src_idx]);
            write_pos += 1;
        }

        if keep_sep {
            let sep_idx = real_len - 1;
            all_input_ids[write_pos] = ids[sep_idx] as i64;
            all_attention_mask[write_pos] = masks[sep_idx] as i64;
            all_token_type_ids[write_pos] = type_ids[sep_idx] as i64;
            token_ids_vec.push(ids[sep_idx]);
        }

        all_token_ids.push(token_ids_vec);
    }

    Ok(PreparedDocumentBatch {
        batch_size,
        batch_max_len,
        all_input_ids,
        all_attention_mask,
        all_token_type_ids: if config.uses_token_type_ids {
            Some(all_token_type_ids)
        } else {
            None
        },
        all_token_ids,
        original_lengths,
        is_query,
        filter_skiplist,
    })
}

fn resolve_prefix_token_id(
    tokenizer: &Tokenizer,
    config: &ColbertConfig,
    is_query: bool,
) -> Result<(u32, usize)> {
    let (prefix_str, prefix_token_id_opt, max_length) = if is_query {
        (
            config.query_prefix.as_str(),
            config.query_prefix_id,
            config.query_length,
        )
    } else {
        (
            config.document_prefix.as_str(),
            config.document_prefix_id,
            config.document_length,
        )
    };

    let prefix_token_id = match prefix_token_id_opt {
        Some(id) => id,
        None => tokenizer.token_to_id(prefix_str).ok_or_else(|| {
            anyhow::anyhow!(
                "Prefix token '{}' not found in tokenizer vocabulary",
                prefix_str
            )
        })?,
    };

    Ok((prefix_token_id, max_length))
}

fn default_fill_values(config: &ColbertConfig, is_query: bool) -> (i64, i64) {
    if is_query && config.do_query_expansion {
        (config.mask_token_id as i64, 1)
    } else {
        (config.pad_token_id as i64, 0)
    }
}

fn content_prefix_plan(real_len: usize, truncate_limit: usize) -> (usize, bool) {
    if real_len > truncate_limit {
        (truncate_limit.saturating_sub(1), true)
    } else {
        (real_len, false)
    }
}

fn final_length(real_len: usize, max_length: usize, keep_sep: bool) -> usize {
    if keep_sep {
        max_length
    } else {
        real_len + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tokenizers::models::wordpiece::WordPiece;
    use tokenizers::Tokenizer;

    fn test_tokenizer() -> Tokenizer {
        let vocab = [
            ("[UNK]".to_string(), 0),
            ("[CLS]".to_string(), 1),
            ("[SEP]".to_string(), 2),
            ("[PAD]".to_string(), 3),
            ("[MASK]".to_string(), 4),
            ("[Q] ".to_string(), 5),
            ("[D] ".to_string(), 6),
            ("hello".to_string(), 7),
        ];

        Tokenizer::new(
            WordPiece::builder()
                .vocab(vocab.into())
                .unk_token("[UNK]".to_string())
                .build()
                .unwrap(),
        )
    }

    fn test_encoding(ids: Vec<u32>, type_ids: Vec<u32>, attention_mask: Vec<u32>) -> Encoding {
        let len = ids.len();
        Encoding::new(
            ids,
            type_ids,
            vec![String::new(); len],
            vec![None; len],
            vec![(0, 0); len],
            vec![0; len],
            attention_mask,
            vec![],
            HashMap::new(),
        )
    }

    #[test]
    fn prepare_batch_inserts_prefix_after_cls() {
        let tokenizer = test_tokenizer();
        let config = ColbertConfig::default();
        let encoding = test_encoding(vec![1, 7, 2], vec![0, 0, 0], vec![1, 1, 1]);

        let prepared = prepare_batch_from_tokenizer_encodings(
            &tokenizer,
            &config,
            vec![encoding],
            true,
            false,
        )
        .unwrap();

        assert_eq!(prepared.batch_max_len, 48);
        assert_eq!(&prepared.all_input_ids[..4], &[1, 5, 7, 2]);
        assert_eq!(&prepared.all_attention_mask[..4], &[1, 1, 1, 1]);
        assert_eq!(prepared.all_token_ids[0], vec![1, 5, 7, 2]);
    }

    #[test]
    fn query_expansion_uses_mask_fill_and_active_attention() {
        let tokenizer = test_tokenizer();
        let mut config = ColbertConfig::default();
        update_token_ids(&mut config, &tokenizer);
        let encoding = test_encoding(vec![1, 7, 2], vec![0, 0, 0], vec![1, 1, 1]);

        let prepared = prepare_batch_from_tokenizer_encodings(
            &tokenizer,
            &config,
            vec![encoding],
            true,
            false,
        )
        .unwrap();

        assert_eq!(prepared.all_input_ids[4], 4);
        assert_eq!(prepared.all_attention_mask[4], 1);
        assert!(prepared.all_token_type_ids.is_some());
    }

    #[test]
    fn token_type_ids_can_be_omitted() {
        let tokenizer = test_tokenizer();
        let mut config = ColbertConfig::default();
        config.uses_token_type_ids = false;
        config.do_query_expansion = false;
        config.query_length = 8;
        let encoding = test_encoding(vec![1, 7, 2], vec![0, 0, 0], vec![1, 1, 1]);

        let prepared = prepare_batch_from_tokenizer_encodings(
            &tokenizer,
            &config,
            vec![encoding],
            true,
            false,
        )
        .unwrap();

        assert_eq!(prepared.batch_max_len, 4);
        assert!(prepared.all_token_type_ids.is_none());
    }
}
