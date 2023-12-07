use std::{
    convert::Infallible,
    fs::File,
    io::Read,
    path::Path,
    str::FromStr,
    sync::{Arc, Mutex},
    thread,
};

// these to uses are for logging debug files out for the prompt and the text inferrence result.
#[cfg(debug_assertions)]
use std::io::Write;

use crossbeam::channel::{bounded, Receiver, Sender};
use llm::Model;
use rand::rngs::ThreadRng;
use serde::{Deserialize, Serialize};

use crate::{
    chatlog::{ChatLog, ChatLogItem},
    config::*,
};
use anyhow::{Context, Error as E, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::Tokenizer;

const DEFAULT_NUM_OF_SENTENCE_MATCHES: usize = 3;

const DEFAULT_TEXT_TO_TOKEN_RATIO: f32 = 3.0;
const DEFAULT_MAX_NEW_TOKENS: usize = 150;

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

            if let Some(local_model_path) = &model_config.path {
                // enable gpu offloading here and default behavior
                // seems to offload all layers.
                let mut model_params = llm::ModelParameters::default();
                model_params.use_gpu = config.use_gpu.unwrap_or(false);
                model_params.gpu_layers = model_config.gpu_layer_count;
                model_params.context_size = model_config.context_size;
                model_params.n_gqa = model_config.gqa;

                // load the model, ignoring any loading progress ...
                let model_path = Path::new(local_model_path);
                llm_model = Some(
                    llm::load_dynamic(
                        Some(llm::ModelArchitecture::Llama),
                        model_path,
                        llm::TokenizerSource::Embedded,
                        model_params,
                        |_| {},
                    )
                    .unwrap_or_else(|err| {
                        panic!("Failed to load model from {model_fileorname:?}: {err}")
                    }),
                );
            }

            // now load the embedding model
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
                embedding_engine,
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

                            engine_state.model = None;
                            engine_state.model_config = model_config.clone();
                            log::debug!(
                                "Loading a different model for configuration: {}",
                                cfg_name
                            );

                            if let Some(local_model_path) = &model_config.path {
                                // enable gpu offloading here and default behavior
                                // seems to offload all layers.
                                let mut model_params = llm::ModelParameters::default();
                                model_params.use_gpu = engine_state.config.use_gpu.unwrap_or(false);
                                model_params.gpu_layers = engine_state.model_config.gpu_layer_count;
                                model_params.context_size = engine_state.model_config.context_size;
                                model_params.n_gqa = engine_state.model_config.gqa;

                                // load the model, ignoring any loading progress ...
                                let model_path = Path::new(local_model_path);
                                engine_state.model = Some(
                                    llm::load_dynamic(
                                        Some(llm::ModelArchitecture::Llama),
                                        model_path,
                                        llm::TokenizerSource::Embedded,
                                        model_params,
                                        |_| {},
                                    )
                                    .unwrap_or_else(|err| {
                                        panic!(
                                            "Failed to load model from {model_fileorname:?}: {err}"
                                        )
                                    }),
                                );
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
    model: Option<Box<dyn Model>>,

    // the currently active model configuration
    model_config: ConfiguredLlm,

    // the model config specified on the command line and 'default' config
    default_model_config: ConfiguredLlm,

    // the configuration file for the application
    config: ConfigurationFile,

    // an optional handle to the vector embedding engine
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
        stop_seqs.push(format!("{}: ", context.character.name));
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
            stop_sequence: if self.config.stop_on_display_name { Some(stop_seqs) } else { None },
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
        let mut session_config: llm::InferenceSessionConfig = Default::default();
        session_config.n_batch = self.config.batch_size.unwrap_or(512);
        session_config.n_threads = self.config.thread_count.unwrap_or(8);

        let mut session = self.model.as_mut().unwrap().start_session(session_config);
        let mut inferred_string = String::new();

        let prompt = self.create_prompt_for_chat_input(context);

        // DEBUG WRITE OUT THE PROMPT TO A FILE.
        #[cfg(debug_assertions)]
        {
            let mut raw_file = File::create(".debug.prompt.txt").unwrap();
            let _ = raw_file.write_all(prompt.as_bytes());
        }

        // build the sampler string to build the sampler chain from
        let mut buffer = String::new();
        if let Some(rep_pen) = context.parameters.repeat_penalty {
            buffer.push_str(format!("repetition:penalty={}", rep_pen).as_str());
            if let Some(pen_range) = context.parameters.repeat_penalty_range {
                buffer.push_str(format!(":last_n={}", pen_range).as_str());
            }
        }

        // a lot of sampler parameters are not valid for mirostat, so handle them separately
        if let Some(mirostat_type) = context.parameters.mirostat {
            // only valid options are 1 and 2
            if mirostat_type == 1 || mirostat_type == 2 {
                buffer.push_str(format!(" mirostat{}", mirostat_type).as_str());

                if let Some(eta) = context.parameters.mirostat_eta {
                    buffer.push_str(format!(":eta={}", eta).as_str());
                }
                if let Some(tau) = context.parameters.mirostat_tau {
                    buffer.push_str(format!(":tau={}", tau).as_str());
                }
            }
        } else {
            if let Some(top_k) = context.parameters.top_k {
                buffer.push_str(format!(" top_k:k={}:min_keep=1", top_k).as_str());
            }
            if let Some(top_p) = context.parameters.top_p {
                buffer.push_str(format!(" top_p:p={}:min_keep=1", top_p).as_str());
            }
            if let Some(min_p) = context.parameters.min_p {
                buffer.push_str(format!(" min_p:p={}:min_keep=1", min_p).as_str());
            }
            if let Some(temperature) = context.parameters.temperature {
                buffer.push_str(format!(" temperature:temperature={}", temperature).as_str());
            }
        }

        let samplers_p = llm::samplers::ConfiguredSamplers::from_str(buffer.as_str());
        let samplers = if let Ok(samplers) = samplers_p {
            samplers.builder.into_chain()
        } else {
            log::error!(
                "Unable to build sampler chain from the configuration requested ({}): {:?}",
                buffer,
                samplers_p.err()
            );
            llm::samplers::ConfiguredSamplers::default()
                .builder
                .into_chain()
        };

        let parameters = llm::InferenceParameters {
            sampler: Arc::new(Mutex::new(samplers)),
        };

        // pull the new-token count from the config file if specified
        let token_count = self
            .config
            .maximum_new_tokens
            .unwrap_or(DEFAULT_MAX_NEW_TOKENS);

        let local_model_unwrapped = self.model.as_ref().unwrap();
        let res = session.infer::<Infallible>(
            // model to use for text generation
            local_model_unwrapped.as_ref(),
            // randomness provider
            &mut self.rng,
            // the prompt to use for text generation, as well as other
            // inference parameters
            &llm::InferenceRequest {
                prompt: prompt.as_str().into(),
                parameters: &parameters,
                play_back_previous_tokens: false,
                maximum_token_count: Some(token_count),
            },
            // llm::OutputRequest
            &mut Default::default(),
            // output callback
            |t| {
                if let llm::InferenceResponse::InferredToken(token) = t {
                    inferred_string.push_str(token.as_str());
                }
                return Ok(llm::InferenceFeedback::Continue);
            },
        );

        match res {
            Err(err) => log::error!("Error: {err}"),
            Ok(stats) => {
                log::debug!("Inference Stats: {}", stats.to_string());

                // we have a tracker variable for how much text we can squeeze into a token
                // budget. this is the learning tracker based on what the library reports
                // as the token count to the prompt we passed in.

                // Disabled for now...
                // const LEARNING_RATE: f32 = 0.1;
                // let token_ratio = prompt.len() as f32 / stats.prompt_tokens as f32;
                // let old_ratio = self
                //     .config
                //     .text_to_token_ratio_prediction
                //     .unwrap_or(DEFAULT_TEXT_TO_TOKEN_RATIO);
                // let ratio_delta = (token_ratio - old_ratio) * LEARNING_RATE;
                // self.config.text_to_token_ratio_prediction = Some(old_ratio + ratio_delta);
                // log::debug!("Text2Token ratio for prompt: {}", token_ratio);
                // log::debug!(
                //     "Text2Token old ratio: {} ... new delta: {}",
                //     old_ratio,
                //     ratio_delta
                // );
            }
        };

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
            Device::Cpu
        } else {
            Device::new_cuda(0).unwrap()
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

    fn build_all_vector_embeddings(
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
                    < (self.config.token_cutoff_limit as f32 * DEFAULT_TEXT_TO_TOKEN_RATIO) as usize
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
    fn get_sentence_similarity_for_last(
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
    device: &Device,
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
