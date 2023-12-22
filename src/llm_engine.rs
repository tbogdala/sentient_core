use std::thread;

// these to uses are for logging debug files out for the prompt and the text inferrence result.
#[cfg(debug_assertions)]
use std::fs::File;
#[cfg(debug_assertions)]
use std::io::Write;

use crossbeam::channel::{bounded, Receiver, Sender};
use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};
use rand::{rngs::ThreadRng, Rng};
use serde::{Deserialize, Serialize};

use crate::{chatlog::ChatLog, config::*};
use anyhow::Context;

#[cfg(feature = "sentence_similarity")]
use crate::vector_embedding_engine::VectorEmbeddingEngine;

#[cfg(feature = "sentence_similarity")]
pub const DEFAULT_NUM_OF_SENTENCE_MATCHES: usize = 3;

pub const DEFAULT_TEXT_TO_TOKEN_RATIO: f32 = 3.0;
pub const DEFAULT_MAX_NEW_TOKENS: usize = 150;
pub const DEFAULT_BATCH_SIZE: usize = 8;
pub const DEFAULT_THREAD_COUNT: usize = 8;

#[derive(Clone, PartialEq)]
pub enum LlmEngineRequest {
    TextInference(TextInferenceContext),
    ImmediateShutdown,
}

#[derive(Clone, PartialEq)]
pub enum LlmEngineResponse {
    NewText(Option<String>, TextInferenceContext),
    ModelLoaded,
}

pub struct LlmEngine {
    pub send_to_server: Sender<LlmEngineRequest>,
    pub recv_on_client: Receiver<LlmEngineResponse>,
    pub handle: thread::JoinHandle<()>,
}
impl LlmEngine {
    pub fn spawn(config: ConfigurationFile, model_fileorname: String) -> LlmEngine {
        let (send_to_server, recv_on_server) = bounded::<LlmEngineRequest>(10);
        let (send_to_client, recv_on_client) = bounded::<LlmEngineResponse>(10);
        let thread_handle = thread::spawn(move || {
            // failures should have been detected before this gets here
            let model_config = config
                .find_model_configuration(&model_fileorname)
                .context("Attempting to find the model name provided in the configuration")
                .unwrap();
            let mut llm_model = None;

            // setup the thread rng
            let mut rng = rand::thread_rng();

            // if we're using a local model, load it up
            if let Some(local_model_path) = &model_config.path {
                // use a provided seed for the model or make a new one
                let this_seed = match model_config.seed {
                    Some(s) => s,
                    None => rng.gen_range(0..i32::MAX),
                };

                let model_params = ModelOptions {
                    context_size: model_config.context_size as i32,
                    seed: this_seed,
                    n_gpu_layers: if config.use_gpu.unwrap_or(false) {
                        model_config.gpu_layer_count.unwrap_or(0) as i32
                    } else {
                        0
                    },
                    n_batch: config.batch_size.unwrap_or(DEFAULT_BATCH_SIZE) as i32,
                    ..Default::default()
                };

                llm_model = match LLama::new(local_model_path.clone(), &model_params) {
                    Ok(m) => Some(m),
                    Err(err) => panic!("Failed to load model from {local_model_path}: {err}"),
                };
            }

            // now load the embedding model
            #[cfg(feature = "sentence_similarity")]
            let embedding_engine = match &config.embedding_model {
                Some(embedding_config) => Some(
                    VectorEmbeddingEngine::new(&embedding_config)
                        .unwrap_or_else(|err| panic!("Failed to load the embedding model: {err}")),
                ),
                None => None,
            };

            // setup a state object
            let mut engine_state = EngineState {
                model: llm_model,
                model_config: model_config.clone(),
                default_model_config: model_config,
                config,

                #[cfg(feature = "sentence_similarity")]
                embedding_engine: embedding_engine,

                rng: rand::thread_rng(),
            };

            // tell the main thread that we've loaded.
            send_to_client
                .send(LlmEngineResponse::ModelLoaded)
                .expect("Failed to acknowledge initial model load sucess.");

            loop {
                // BLOCK UNTIL NEW REQUEST
                let result;
                let request = recv_on_server.recv().unwrap_or_else(|err| {
                    panic!("LlmEngine thread's recv failed: {}", err);
                });

                match request {
                    LlmEngineRequest::ImmediateShutdown => {
                        return;
                    }
                    LlmEngineRequest::TextInference(context) => {
                        let mut new_context = context;

                        let cfg_to_load = match &new_context.model_config_override {
                            Some(model_config_ovr)
                                if !engine_state.model_config.name.eq(model_config_ovr) =>
                            {
                                Some(model_config_ovr.to_owned())
                            }
                            None if !engine_state
                                .model_config
                                .name
                                .eq(&engine_state.default_model_config.name) =>
                            {
                                Some(engine_state.default_model_config.name.to_owned())
                            }
                            _ => None,
                        };
                        // need to load up a different model
                        if let Some(cfg_name) = cfg_to_load {
                            // TODO: this is a dupe of above logic, mostly; refactor at some point
                            // failures should have been detected before this gets here
                            let model_config = engine_state.config
                                .find_model_configuration(&cfg_name)
                                .context("Attempting to find the model name provided in the configuration on text inferrence request")
                                .unwrap();

                            // free the model so we got memory to load the next one
                            if let Some(model) = engine_state.model.as_mut() {
                                model.free_model();
                                engine_state.model = None;
                            }
                            engine_state.model_config = model_config.clone();
                            log::debug!(
                                "Loading a different model for configuration: {}",
                                cfg_name
                            );

                            if let Some(local_model_path) = &model_config.path {
                                // use a provided seed for the model or make a new one
                                let this_seed = match model_config.seed {
                                    Some(s) => s,
                                    None => engine_state.rng.gen_range(0..i32::MAX),
                                };

                                let model_params = ModelOptions {
                                    context_size: model_config.context_size as i32,
                                    seed: this_seed,
                                    n_gpu_layers: if engine_state.config.use_gpu.unwrap_or(false) {
                                        model_config.gpu_layer_count.unwrap_or(0) as i32
                                    } else {
                                        0
                                    },
                                    n_batch: engine_state
                                        .config
                                        .batch_size
                                        .unwrap_or(DEFAULT_BATCH_SIZE)
                                        as i32,
                                    ..Default::default()
                                };

                                engine_state.model =
                                    match LLama::new(local_model_path.clone(), &model_params) {
                                        Ok(m) => Some(m),
                                        Err(err) => panic!(
                                            "Failed to load model from {local_model_path}: {err}"
                                        ),
                                    };
                            }
                        }

                        // if we have a local llm model loaded use that, otherwise try remote API config
                        let new_text = if !engine_state.model_config.path.is_none() {
                            engine_state.text_infer(&mut new_context)
                        } else {
                            engine_state.text_infer_kobold(&mut new_context)
                        };
                        result = LlmEngineResponse::NewText(new_text, new_context);
                    }
                };

                // SEND THE RESULT FROM THE SERVER
                if let Err(err) = send_to_client.send(result) {
                    log::error!("LlmEngine thread's send failed: {}", err);
                }
                log::trace!("One job-cycle complete in the llm engine thread.");
            }
        });

        return LlmEngine {
            send_to_server,
            recv_on_client,
            handle: thread_handle,
        };
    }
}

#[derive(Clone, PartialEq)]
pub struct TextInferenceContext {
    pub character: CharacterFileYaml,

    // the name of the model configuration to use for this text generation request
    pub model_config_override: Option<String>,

    // because other_participants, when generating will be set to 'character', we'll need
    // this member to remember who the 'main' character is.
    pub chatlog_owner: CharacterFileYaml,

    pub other_participants: Vec<(CharacterFileYaml, Option<String>)>,

    pub chatlog: ChatLog,

    // set to true if inference should try and continue the last line of the chain
    pub should_continue: bool,

    pub parameters: ConfiguredParameters,
}

struct EngineState {
    // the loaded model
    model: Option<LLama>,

    // the currently active model configuration
    model_config: ConfiguredLlm,

    // the model config specified on the command line and 'default' config
    default_model_config: ConfiguredLlm,

    // the configuration file for the application
    config: ConfigurationFile,

    // an optional handle to the vector embedding engine
    #[cfg(feature = "sentence_similarity")]
    embedding_engine: Option<VectorEmbeddingEngine>,

    // our thread random generator
    rng: ThreadRng,
}
impl EngineState {
    // given the string a user inputs, turn that into the whole
    // prompt that is given to the engine
    fn create_prompt_for_chat_input(&self, context: &mut TextInferenceContext) -> String {
        // and then create the system message with the context for the bot
        let mut buf = String::new();
        buf.push_str(self.model_config.prompt_instruct_template.as_str());

        // order of operations is important here so that the names are replaced last.
        buf = buf.replace("<|character_description|>", &context.character.description);
        buf = buf.replace("<|current_context|>", &context.chatlog.current_context);
        if let Some(user_desc) = &context.chatlog.user_description {
            buf = buf.replace("<|user_description|>", user_desc);
        }

        // test to see if this template wants the vector embedding support as well
        // only works with non-empty chat logs.
        #[cfg(feature = "sentence_similarity")]
        if buf.contains("<|similar_sentences|>") && context.chatlog.len() > 0 {
            if let Some(embedding_engine) = &self.embedding_engine {
                // make sure all the chat log has their embeddings calculated
                embedding_engine.build_all_vector_embeddings(&mut context.chatlog, false);

                let requested_match_count = self
                    .model_config
                    .similar_sentence_count
                    .unwrap_or(DEFAULT_NUM_OF_SENTENCE_MATCHES);
                let end_offset = if context.should_continue { 1 } else { 0 };
                let matches = embedding_engine.get_sentence_similarity_for_last(
                    &context.chatlog,
                    end_offset,
                    requested_match_count,
                );
                let matched_strings: Vec<String> = matches.iter().map(|m| m.2.to_owned()).collect();
                let joined_matches = matched_strings.join("\n");
                buf = buf.replace("<|similar_sentences|>", joined_matches.as_str());
            } else {
                log::warn!("The LLM prompt includes <|similar_sentences|> but an embedding model wasn't configured, so it's being skipped.");
                buf = buf.replace("<|similar_sentences|>", "");
            }
        }

        buf = buf.replace("<|character_name|>", &context.character.name);
        buf = buf.replace("<|user_name|>", &self.config.display_name);

        // start off with the string for the request
        let mut history_log = String::new();
        let mut continue_line = String::new();

        // now we reverse walk the conversation chain and stack in more message history

        // get the current ratio used to predict how well text is going to compress down into tokens
        // so that the context memory can get maximized.
        let text2token_ratio: f32 = self
            .config
            .text_to_token_ratio_prediction
            .unwrap_or(DEFAULT_TEXT_TO_TOKEN_RATIO);

        // pull the requested max new token count from the configuration
        let token_count = self
            .config
            .maximum_new_tokens
            .unwrap_or(DEFAULT_MAX_NEW_TOKENS);

        // figure out our remaining token budget in text characters and build a history log based on that.
        let prompt_limit: usize = ((self.model_config.context_size - token_count) as f32
            * text2token_ratio) as usize
            - buf.len();
        for conv_turn in context.chatlog.iter().rev() {
            let turn_str = conv_turn.get_name_and_items_as_string();

            // if we're continuing a response and haven't pulled the log item to continue
            // do that here - should trigger on the first iteration.
            if context.should_continue && continue_line.is_empty() {
                // remove the name from the last log line if it's there ... in multiline responses it may not be.
                if turn_str.starts_with(&context.character.name) {
                    continue_line = turn_str[context.character.name.len() + 1..].to_owned();
                } else {
                    continue_line = turn_str.to_owned();
                }
            } else {
                let new_history = format!("{}\n{}", turn_str, history_log);
                if new_history.len() + continue_line.len() >= prompt_limit {
                    break;
                }
                history_log = new_history;
            }
        }

        buf = buf.replace("<|chat_history|>", history_log.trim_end());

        // This theoretically should be the last thing added since it's the line getting continued
        if !continue_line.is_empty() {
            buf.push_str(&continue_line);
        }

        return buf;
    }

    fn text_infer_kobold(&mut self, context: &mut TextInferenceContext) -> Option<String> {
        // build the prompt
        let prompt = self.create_prompt_for_chat_input(context);

        // DEBUG WRITE OUT THE PROMPT TO A FILE.
        #[cfg(debug_assertions)]
        {
            let mut raw_file = File::create(".debug.prompt.txt").unwrap();
            let _ = raw_file.write_all(prompt.as_bytes());
        }

        // Use a default 120 minute timeout, unless configured otherwise
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(
                self.model_config.remote_timeout_s.unwrap_or(60 * 120),
            ))
            .build()
            .expect("Failed to create the blocking reqwest client for KoboldAPI.");

        // If not supplied we try to use the localhost
        let api_host = match self.model_config.remote_server.as_ref() {
            Some(s) => s,
            None => {
                log::warn!("KoboldAPI: currently selected model didn't specify 'remote_server'; defaulting to 'http://localhost:5001'");
                "http://localhost:5001"
            }
        };

        // build an array of character names to stop on for everyone
        let mut stop_seqs = vec![format!("{}: ", self.config.display_name)];
        stop_seqs.push(format!("{}: ", context.chatlog_owner.name));
        if !context.other_participants.is_empty() {
            for other in &context.other_participants {
                stop_seqs.push(format!("{}: ", other.0.name));
            }
        }

        let textgen_url = format!("{}{}", api_host, "/api/v1/generate");
        let textgen_request = TextgenRemoteRequestKobold {
            prompt,
            max_context_length: Some(self.model_config.context_size),
            max_length: self.config.maximum_new_tokens,
            temperature: context.parameters.temperature,
            top_k: context.parameters.top_k,
            top_p: context.parameters.top_p,
            min_p: context.parameters.min_p,
            rep_pen: context.parameters.repeat_penalty,
            rep_pen_range: context.parameters.repeat_penalty_range,
            typical: None,
            sampler_seed: None,
            mirostat: context.parameters.mirostat,
            mirostat_eta: context.parameters.mirostat_eta,
            mirostat_tau: context.parameters.mirostat_tau,
            trim_stop: Some(true),
            stop_sequence: if self.config.stop_on_display_name {
                Some(stop_seqs)
            } else {
                None
            },
        };

        // serialize the request to JSON and send it to the server; blocking because this is all
        // done on a separate thread from the UI anyways, and that usage pattern mirrors how
        // locally hosted generation works.
        let textgen_request_json = serde_json::to_string(&textgen_request).expect(
            "Failed to serialize the KoboldAPI parameters for the text generation request.",
        );
        let textgen_resp = client
            .post(&textgen_url)
            .body(textgen_request_json)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header(reqwest::header::ACCEPT, "application/json")
            .send()
            .expect("KoboldAPI call failed for generating text from a prompt");
        if textgen_resp.status() != reqwest::StatusCode::OK {
            log::error!(
                "KoboldAPI: Failed to generate text for the given prompt. Status: {}",
                textgen_resp.status()
            );
            return None;
        }

        let textgen_resp_text = textgen_resp
            .text()
            .expect("KoboldAPI: Failed to get the JSON from the text generation response body.");
        let textgen_resp: TextgenResponseBodyKobold = serde_json::from_str(&textgen_resp_text)
            .expect(
                "KoboldAPI: Failed to deserialize the JSON from the text generation response body.",
            );
        if textgen_resp.results.is_empty() {
            log::error!("KoboldAPI: Failed to generate text for the given prompt. Empty result was returned.");
            return None;
        }

        let mut inferred_string = textgen_resp.results[0].text.clone();

        // DEBUG WRITE OUT THE PROMPT TO A FILE.
        #[cfg(debug_assertions)]
        {
            let mut raw_file = File::create(".debug.result.txt").unwrap();
            let _ = raw_file.write_all(inferred_string.as_bytes());
        }

        // if enabled, stop the inferred string at any detected name of a participant.
        if self.config.stop_on_display_name {
            self.split_inference_at_display_names(context, &mut inferred_string);
        }

        Some(inferred_string)
    }

    fn text_infer(&mut self, context: &mut TextInferenceContext) -> Option<String> {
        let this_seed = match self.model_config.seed {
            Some(s) => s,
            None => -1, // this should make llama.cpp make a random seed
        };

        let mut predict_options = PredictOptions {
            seed: this_seed,
            batch: self.config.batch_size.unwrap_or(DEFAULT_BATCH_SIZE) as i32,
            threads: self.config.thread_count.unwrap_or(DEFAULT_THREAD_COUNT) as i32,
            tokens: self
                .config
                .maximum_new_tokens
                .unwrap_or(DEFAULT_MAX_NEW_TOKENS) as i32,
            ..Default::default()
        };

        // Setup all the sampler options, overriding the defaults presented by
        // the library if they're configured in the parameter set.
        if let Some(mirostat_type) = context.parameters.mirostat {
            // only valid options are 1 and 2
            if mirostat_type == 1 || mirostat_type == 2 {
                // disable top_p / top_k / min_p / temp
                predict_options.top_k = 0;
                predict_options.top_p = 1.0;
                predict_options.temperature = 1.0;
                predict_options.min_p = 0.0;
                predict_options.mirostat = mirostat_type as i32;
                if let Some(eta) = context.parameters.mirostat_eta {
                    predict_options.mirostat_eta = eta;
                }
                if let Some(tau) = context.parameters.mirostat_tau {
                    predict_options.mirostat_tau = tau;
                }
            }
        } else {
            predict_options.mirostat = 0;
            if let Some(top_k) = context.parameters.top_k {
                predict_options.top_k = top_k as i32;
            }
            if let Some(top_p) = context.parameters.top_p {
                predict_options.top_p = top_p;
            }
            if let Some(min_p) = context.parameters.min_p {
                predict_options.min_p = min_p;
            }
            if let Some(temp) = context.parameters.temperature {
                predict_options.temperature = temp;
            }
        }
        if let Some(rep_pen) = context.parameters.repeat_penalty {
            predict_options.penalty = rep_pen;
        }
        if let Some(rep_range) = context.parameters.repeat_penalty_range {
            predict_options.repeat = rep_range as i32;
        }

        let prompt = self.create_prompt_for_chat_input(context);

        // DEBUG WRITE OUT THE PROMPT TO A FILE.
        #[cfg(debug_assertions)]
        {
            let mut raw_file = File::create(".debug.prompt.txt").unwrap();
            let _ = raw_file.write_all(prompt.as_bytes());
        }

        let local_model_unwrapped = self.model.as_ref().unwrap();
        let (mut inferred_string, timings) =
            match local_model_unwrapped.predict(prompt, predict_options) {
                Ok((s, t)) => (s, t),
                Err(err) => {
                    log::error!("Text inference failed: {}", err);
                    return None;
                }
            };

        log::debug!("{} tokens ; load {:.2}ms ; sample {:.2}T/s ; prompt ({}) eval {:.2}T/s ; eval {:.2}T/s ; total {:.2} ms ({:.2} T/s)",
            timings.n_eval,
            timings.t_load_ms,
            1e3 / timings.t_sample_ms * timings.n_sample as f64,
            timings.n_p_eval,
            1e3 / timings.t_p_eval_ms * timings.n_p_eval as f64,
            1e3 / timings.t_eval_ms * timings.n_eval as f64,
            timings.t_end_ms - timings.t_start_ms,
            1e3 / (timings.t_end_ms - timings.t_start_ms) * timings.n_eval as f64
            );

        // DEBUG WRITE OUT THE PROMPT TO A FILE.
        #[cfg(debug_assertions)]
        {
            let mut raw_file = File::create(".debug.result.txt").unwrap();
            let _ = raw_file.write_all(inferred_string.as_bytes());
        }

        // TODO: Actually do the stopping of the token generation in the above loop instead.
        // if enabled, stop the inferred string at any detected name of a participant.
        if self.config.stop_on_display_name {
            self.split_inference_at_display_names(context, &mut inferred_string);
        }

        return Some(inferred_string);
    }

    // the purpose of this function is to split the response away from the part where
    // it might try to generate a response for another participant.
    fn split_inference_at_display_names(
        &self,
        context: &TextInferenceContext,
        inferred_string: &mut String,
    ) {
        let mut earliest = None;

        // this is a little sloppy but should work. check user first
        let stop_phrase = format!("{}:", self.config.display_name);
        if let Some(found) = inferred_string.find(&stop_phrase) {
            let prev_earliest = earliest.unwrap_or(inferred_string.len());
            if found < prev_earliest {
                earliest = Some(found);
            }
        }

        // check the character name that's doing the generation
        let stop_phrase = format!("{}:", context.character.name);
        if let Some(found) = inferred_string.find(&stop_phrase) {
            let prev_earliest = earliest.unwrap_or(inferred_string.len());
            if found < prev_earliest {
                earliest = Some(found);
            }
        }

        // the main character wont be listed as an 'other_participant' when the text
        // inference request is created, so we check here to see if the chatlog
        // owner is different than the current character generating text and if so
        // we look to find the original owner's name too
        if !context
            .character
            .name
            .eq_ignore_ascii_case(&context.chatlog_owner.name)
        {
            let stop_phrase = format!("{}:", context.chatlog_owner.name);
            if let Some(found) = inferred_string.find(&stop_phrase) {
                let prev_earliest = earliest.unwrap_or(inferred_string.len());
                if found < prev_earliest {
                    earliest = Some(found);
                }
            }
        }

        // check for the name of any other participants
        for other in context.other_participants.iter() {
            let stop_phrase = format!("{}:", other.0.name);
            if let Some(found) = inferred_string.find(&stop_phrase) {
                let prev_earliest = earliest.unwrap_or(inferred_string.len());
                if found < prev_earliest {
                    earliest = Some(found);
                }
            }
        }

        if let Some(earliest) = earliest {
            log::debug!(
                "Splitting off response at {}\n{}",
                earliest,
                inferred_string
            );
            let _ = inferred_string.split_off(earliest); // we discard the rest
        }
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct TextgenRemoteRequestKobold {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_context_length: Option<usize>, // max number of tokens to send to model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_length: Option<usize>, // number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rep_pen: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rep_pen_range: Option<usize>,
    // sampler_order
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampler_seed: Option<i64>,
    // stop_sequence
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    // tfs
    // top_a
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typical: Option<f32>,
    // use_default_badwordsids
    #[serde(skip_serializing_if = "Option::is_none")]
    mirostat: Option<usize>, // 0, 1 or 2
    #[serde(skip_serializing_if = "Option::is_none")]
    mirostat_tau: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mirostat_eta: Option<f32>,
    // genkey
    // grammar
    // grammar_retain_state
    // memory
    #[serde(skip_serializing_if = "Option::is_none")]
    trim_stop: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequence: Option<Vec<String>>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct TextgenResponseBodyKobold {
    results: Vec<TextgenResponseBodyResultKobold>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct TextgenResponseBodyResultKobold {
    text: String,
}
