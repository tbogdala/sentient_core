use anyhow::{Context, Error as E, Result};
use std::{fs::File, io::Read, path::Path};

use candle_core::Tensor;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::Tokenizer;

use crate::{
    chatlog::{ChatLog, ChatLogItem},
    config::ConfiguredEmbeddingModel,
};

pub struct VectorEmbeddingEngine {
    model: BertModel,
    tokenizer: Tokenizer,
    config: ConfiguredEmbeddingModel,
}
impl VectorEmbeddingEngine {
    // creates a new VectorEmbedingEngine and gets it ready to generate embeddings.
    //
    // emb_model_dir should be a directory that contains: `config.json`, `tokenizer.json`, `model.safetensors`
    // for the BERT embedding model.
    // token_cutoff_limit should be the number of incoming tokens the embedding model can proces before
    // it clips the input. (commonly 256 or 512)
    pub fn new(emb_config: &ConfiguredEmbeddingModel) -> Result<Self> {
        //emb_model_dir: &str, token_cutoff_limit: usize
        let emb_model_dir = &emb_config.dir_path;

        let device = if emb_config.use_cpu {
            candle_core::Device::Cpu
        } else {
            candle_core::Device::new_cuda(0).unwrap()
        };

        let config_filename = format!("{}/config.json", emb_model_dir);
        let tokenizer_filename = format!("{}/tokenizer.json", emb_model_dir);

        let config_str = std::fs::read_to_string(config_filename)
            .context("Attempting to read config.json for the embedding model")?;
        let config: Config = serde_json::from_str(&config_str)
            .context("Attempting to deserialize config.json for the embedding model")?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(E::msg)
            .unwrap();
        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        // attempt to load the safetensor model filename first but fallback to the pth format if needed
        let weights_filename_st = format!("{}/model.safetensors", emb_model_dir);
        let safetensor_path = Path::new(&weights_filename_st);
        let vb = if safetensor_path.exists() {
            let mut weights_bytes = Vec::new();
            let mut weights_file = File::open(safetensor_path)
                .context("Attempting to open model.safetensors for the embedding model")?;
            weights_file
                .read_to_end(&mut weights_bytes)
                .context("Attempting to read model.safetensors for the embedding model")?;
            candle_nn::VarBuilder::from_buffered_safetensors(weights_bytes, DTYPE, &device)
                .context("Processing safetensor weights for the embedding model.")?
        } else {
            let weights_filename_pth = format!("{}/pytorch_model.bin", emb_model_dir);
            candle_nn::VarBuilder::from_pth(weights_filename_pth, DTYPE, &device)
                .context("Processing pth weights for the embedding model.")?
        };

        let model = BertModel::load(vb, &config).context("Attempting to build the BERT model")?;

        Ok(Self {
            model,
            tokenizer,
            config: emb_config.clone(),
        })
    }

    pub fn build_all_vector_embeddings(
        &self,
        // the chatlog to build embeddings for
        chatlog: &mut ChatLog,
        // if false it will skip chatlogitems with non-empty embedding vectors
        force_recalculation: bool,
    ) {
        // let mut chatlog_embeddings: Vec<Tensor> = Vec::new();
        let device = &self.model.device;
        for i in 0..chatlog.len() {
            let chatlogitem: &mut ChatLogItem = chatlog.get_mut(i).unwrap();
            // if we're not forcing recalculation and we already have embeddings, move on...
            if chatlogitem.embeddings.is_empty() == false && force_recalculation == false {
                continue;
            }

            // get the whole text of the chat log item so that we can do embeddings on sentence boundaries
            let whole_text = chatlogitem.get_name_and_items_as_string();

            let mut chunked_line = Vec::new();
            let mut buffer = String::new();
            for line in whole_text.lines() {
                // first check to see if we can add new line to buffer without overflowing our token budget
                if buffer.len() + line.len()
                    < (self.config.token_cutoff_limit as f32
                        * crate::llm_engine::DEFAULT_TEXT_TO_TOKEN_RATIO)
                        as usize
                {
                    buffer.push_str(line);
                } else {
                    // we can't fit this sentence, but handle a special case where buffer is empty and this
                    // is the first sentence - which must be ungodly long - so it's just gonna have to get
                    // truncated by the embedding model.
                    if buffer.is_empty() {
                        buffer.push_str(line);
                    }

                    // so now we know we're maxed out for our budget; move the buffer to the vector of
                    // chunked lines and clear it out for a new chunk start.
                    chunked_line.push(buffer);
                    buffer = String::new();
                }
            }

            // any remaining buffer gets turned into a chunk
            chunked_line.push(buffer);

            // now we go through and make embeddings for each chunk
            let embedding_encode_pretext = match &self.config.encode_pretext {
                Some(s) => s.as_str(),
                None => "",
            };
            chatlogitem.embeddings.clear();
            for line in &chunked_line {
                match generate_vector_embedding(
                    device,
                    &self.model,
                    &self.tokenizer,
                    embedding_encode_pretext,
                    line,
                ) {
                    Ok(embedding) => {
                        log::trace!(
                            "Loaded and encoded sentence {i} (shape {:?})...",
                            embedding.shape()
                        );
                        chatlogitem.embeddings.push(embedding);
                    }
                    Err(err) => {
                        log::error!(
                            "Failed to encode vector embeddings for sentence {i}: {}",
                            err
                        );
                    }
                }
            }
        }
    }

    // returns the number of requested similarities, if possible, as a vector of tuples
    // with each tuple being: index into the chatlog, similarity score, chatlogitem's text.
    // The 'extra_offset' parameter should be 0 by default, but can be increased to further skip
    // messages from the end of the log. (e.g. 'extra_offset' of 1 means that it selects the second to last
    // chatlogitem in the chatlog)
    pub fn get_sentence_similarity_for_last(
        &self,
        chatlog: &ChatLog,
        extra_offset: usize,
        number_requested: usize,
    ) -> Vec<(usize, f32, String)> {
        let mut matches = Vec::new();

        // get the last item to use as a test
        let last_item = chatlog
            .get(0.max(chatlog.len() - 1 - extra_offset))
            .context(
                "Attempting to get last chatlogitem in the log to use for searching embeddings",
            )
            .unwrap();
        log::debug!(
            "About to test for {} similarities for the last log item: {}",
            number_requested,
            last_item.get_name_and_items_as_string()
        );

        let embedding_query_pretext = match &self.config.query_pretext {
            Some(s) => s.as_str(),
            None => "",
        };

        let text = &last_item.get_name_and_items_as_string();
        let device = &self.model.device;

        // Note: This doesn't cope with multiple embeddings needed to cover long similarity tests from an incoming message
        let test_embedding = generate_vector_embedding(
            device,
            &self.model,
            &self.tokenizer,
            embedding_query_pretext,
            text,
        )
        .context("Generating embedding for query in sentence similarity test.")
        .unwrap();

        let mut similarities = vec![];
        for (i, item) in chatlog.iter().take(chatlog.len() - 1).enumerate() {
            for item_embedding in item.embeddings.iter() {
                match vector_embedding_cosine_similarity(&test_embedding, item_embedding) {
                    Ok(cosine_similarity) => similarities.push((cosine_similarity, i)),
                    Err(err) => log::error!(
                        "Failed to encode vector embeddings for sentence {i}: {}",
                        err
                    ),
                }
            }
        }

        let num_to_get = if number_requested > similarities.len() {
            similarities.len()
        } else {
            number_requested
        };
        similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
        for &(score, i) in similarities[..num_to_get].iter() {
            let matched_item = chatlog.get(i).unwrap();
            let result_str = matched_item.get_name_and_items_as_string();
            log::debug!("Result #{i} Score:{score:.2} Text: {}", result_str);
            matches.push((i, score, result_str));
        }

        matches
    }
}

// generates a vector embedding Tensor with the device, model and tokenizer passed in for the text specified.
fn generate_vector_embedding(
    device: &candle_core::Device,
    model: &BertModel,
    tokenizer: &Tokenizer,
    embedding_pretext: &str,
    text: &str,
) -> Result<Tensor> {
    // prepend a directive, if appropriate for the embedding model
    let embedding_text = [embedding_pretext, text].concat();

    let tokens = tokenizer
        .encode(embedding_text, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    let token_type_ids = token_ids.zeros_like()?;
    let ys = model.forward(&token_ids, &token_type_ids)?;

    // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
    let (_n_sentence, n_tokens, _hidden_size) = ys.dims3()?;
    let embedding = (ys.sum(1)? / (n_tokens as f64))?.squeeze(0)?;

    // L2 normalization ripped from Candle example - not important with cosine similarity
    // let normalized = embedding.broadcast_div(&embedding.sqr()?.sum_keepdim(0)?.sqrt()?)?;

    Ok(embedding)
}

// calculates the cosine similarity between two vector embedding Tensors
fn vector_embedding_cosine_similarity(first: &Tensor, second: &Tensor) -> Result<f32> {
    let sum_ij = (second * first)?.sum_all()?.to_scalar::<f32>()?;
    let sum_i2 = (second * second)?.sum_all()?.to_scalar::<f32>()?;
    let sum_j2 = (first * first)?.sum_all()?.to_scalar::<f32>()?;

    Ok(sum_ij / (sum_i2 * sum_j2).sqrt())
}
