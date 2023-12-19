use std::path::{Path, PathBuf};

use directories::BaseDirs;
use ratatui::prelude::Alignment;
use serde::Deserialize;

pub const CURRENT_VERSION: u16 = 1;
pub const APPLICATION_CONFIG_FOLDER_NAME: &str = "sentinel_core";
pub const LOG_FILE_NAME: &str = "log.json";

#[derive(Default, Debug, Clone, PartialEq, Deserialize)]
pub struct CharacterFileYaml {
    // the name of the character as it should show up in the logs and UI
    pub name: String,

    // the optional color for the character's name in the chat UI
    pub name_rgb: Option<[u8; 3]>,

    // the optional color for quoted text from the character in the chat UI
    pub quotes_rgb: Option<[u8; 3]>,

    // the optional color for the regular, non-quoted text from the character in the chat UI
    pub text_rgb: Option<[u8; 3]>,

    // the character description that gets substituted in the prompt template: <|character_description|>
    pub description: String,

    // the log lines to use as default when creating a new chatlog for the character
    pub greeting: String,

    // the starting context of the character, which gets copied to new logs;
    // after that, the chatlog current_context should be used.
    pub context: String,
}
impl CharacterFileYaml {
    pub fn load_character(filepath: &PathBuf) -> CharacterFileYaml {
        // if we found a file, deserialize it as yaml
        match std::fs::read_to_string(filepath) {
            Ok(plain_string) => {
                match serde_yaml::from_str::<CharacterFileYaml>(plain_string.as_str()) {
                    Ok(cfg) => return cfg,
                    Err(err) => {
                        log::error!(
                            "Failed to deserialize the configuration file ({:?}): {}",
                            filepath,
                            err
                        );
                    }
                };
            }
            Err(err) => log::error!("Failed to load the character file ({:?}): {err}", filepath),
        };

        // if we made it here, no config file was found, or if it was found, it could not be deserialized as yaml.
        log::warn!(
            "Using a default configuration file from memory since none were located to be read."
        );
        return Default::default();
    }

    // creates a new vector with the processed template from the character file
    pub fn get_greeting(&self, user_name: &str) -> Vec<String> {
        let mut greeting = Vec::new();
        for line in self.greeting.lines() {
            greeting.push(self.process_string_templates(user_name, &line.to_owned()));
        }
        greeting
    }

    // replaces the associated tags in the character file with the actual values.
    // NOTE: currently supports `<|character_name|>` and `<|user_name|>`.
    fn process_string_templates(&self, user_name: &str, input: &String) -> String {
        input
            .replace("<|character_name|>", &self.name)
            .replace("<|user_name|>", user_name)
    }
}

#[derive(Clone, Default, PartialEq, Deserialize)]
pub enum ConversationTurnName {
    USER,
    #[default]
    BOT,
}
impl std::fmt::Display for ConversationTurnName {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ConversationTurnName::USER => write!(f, "USER"),
            ConversationTurnName::BOT => write!(f, "BOT"),
        }
    }
}

#[derive(Default, Clone, PartialEq, Deserialize)]
pub struct ConversationTurn {
    pub name: ConversationTurnName,
    pub value: String,
}

#[derive(Deserialize, PartialEq, Debug, Default, Clone)]
pub struct ConfiguredLlm {
    // a user-friendly name used to identify the model on the command-line
    pub name: String,

    // the path to the model file to load locally for text inference
    pub path: Option<String>,

    // the remote host name for a server that will perform the text
    // inference instead of doing it locally; currently only Koboldcpp is supported
    pub remote_server: Option<String>,

    // the number of seconds to wait for a server to respond before erroring
    // only applies when using 'remote_server' and not 'path' to load locally
    pub remote_timeout_s: Option<u64>,

    // how much room to budget for a complete context
    pub context_size: usize,

    // the number of similar chat log items to pull up using vector embeddings,
    // which requires a configured vector embedding model in the configuration.
    pub similar_sentence_count: Option<usize>,

    // the number of layers to offload to the gpu.
    // applies only to locally hosted models
    pub gpu_layer_count: Option<usize>,

    // the seed to use for this particular model when generating text
    // if not set, a random one will be chosen
    pub seed: Option<i32>,

    // the string used as the main template for text inference
    // with several tags that get replaced with content at
    // inference time.
    pub prompt_instruct_template: String,
}

#[derive(Deserialize, PartialEq, Debug, Default, Clone)]
pub struct ConfiguredEmbeddingModel {
    // the path to the model folder that should contain the 'config.json',
    // 'tokenizer.json' and 'model.safetensors' BERT model files to use
    // as the vector embedding engine.
    pub dir_path: String,

    // The embedding models have a fixed context size in tokens. This variable
    // will be used to break apart sentences in a way to make sure there is
    // minimal data loss when generating the embeddings.
    pub token_cutoff_limit: usize,

    // By default, the embedding engine will use the GPU. Setting this value to
    // true will force it to use the CPU instead.
    pub use_cpu: bool,

    // Optional pretext string to prepend to the text when using the embedding to
    // query a vector store.
    pub query_pretext: Option<String>,

    // Optional pretext string to prepend to the text when using the embedding to
    // encode text for a vector store.
    pub encode_pretext: Option<String>,
}

#[derive(Deserialize, PartialEq, Debug, Clone)]
pub enum Justification {
    Left,
    Right,
    Center,
}
impl From<Justification> for Alignment {
    fn from(value: Justification) -> Self {
        match value {
            Justification::Center => Alignment::Center,
            Justification::Left => Alignment::Left,
            Justification::Right => Alignment::Right,
        }
    }
}

#[derive(Deserialize, PartialEq, Debug, Clone, Default)]
pub struct ConfiguredParameters {
    pub name: String,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_penalty_range: Option<usize>,
    pub temperature: Option<f32>,

    pub mirostat: Option<usize>, // 0=disabled, 1=mirostat1, 2=mirostat2
    pub mirostat_eta: Option<f32>,
    pub mirostat_tau: Option<f32>,
}

#[derive(Deserialize, PartialEq, Debug, Clone)]
pub struct ConfigurationFile {
    // version number for the file which should be incremented on breaking changes
    pub version: u16,

    // the display name to use for the 'USER' in the chat log. not only does this show up
    // on screen, but this is what gets put in the backend prompts as well.
    pub display_name: String,

    // the color to use for the display name of the 'USER' in the chat log.
    pub display_name_rgb: Option<[u8; 3]>,

    // the color to use for text in quotes of the 'USER' in the chat log.
    pub quotes_rgb: Option<[u8; 3]>,

    // the color to use for the normal text, not quoted, for the 'USER' in the chat log.
    pub text_rgb: Option<[u8; 3]>,

    // the foreground RGB color of the 'primary' element in the progress bar
    pub progress_primary_rgb: Option<[u8; 3]>,

    // the foreground RGB color of the 'secondary' element in the progress bar
    pub progress_secondary_rgb: Option<[u8; 3]>,

    // optional setting to determine how the text should be justified.
    pub chat_text_justification: Option<Justification>,

    // optional setting to add a 'buffer' between chatlog items to aid in visually grouping them.
    pub add_visual_buffer_between_chatlog_items: Option<bool>,

    // if true, this will trim the text inferrence to just before the first usage of " {display_name}:"
    pub stop_on_display_name: bool,

    // the current prediction multiplier representing the mount of text characters per token, on average,
    // after tokenization. used to predict how much can be added to the chat history buff and still keep
    // the requested token window size open.
    pub text_to_token_ratio_prediction: Option<f32>,

    // a suggestion of the number of tokens that can be returned by the llm
    pub maximum_new_tokens: Option<usize>,

    // whether or not to use GPU accelleration; must also be configured right in Cargo.toml
    pub use_gpu: Option<bool>,

    // should be set to the number of cores in your cpu
    pub thread_count: Option<usize>,

    // defaults to 8; set to 256 or 512 when `use_gpu` is set -- this will drastically improve
    // performance if your vram budget allows for it.
    pub batch_size: Option<usize>,

    // a vector of hyperparameter sets to use for controlling text inferrence.
    pub parameters: Vec<ConfiguredParameters>,

    // the list of configured models
    pub models: Vec<ConfiguredLlm>,

    pub embedding_model: Option<ConfiguredEmbeddingModel>,
}

impl Default for ConfigurationFile {
    fn default() -> Self {
        return ConfigurationFile {
            version: CURRENT_VERSION,
            display_name: "USER".to_owned(),
            display_name_rgb: None,
            quotes_rgb: None,
            text_rgb: None,
            chat_text_justification: None,
            progress_primary_rgb: None,
            progress_secondary_rgb: None,
            text_to_token_ratio_prediction: None,
            maximum_new_tokens: None,
            use_gpu: Some(false),
            thread_count: Some(8),
            batch_size: Some(512),
            add_visual_buffer_between_chatlog_items: None,
            stop_on_display_name: true,
            parameters: Vec::new(),
            models: Vec::new(),
            embedding_model: None,
        };
    }
}

impl ConfigurationFile {
    // loads the configuration file by using the alternative path specified or by searching
    // common locations for the config file to load.
    // if those fail to find a file, then a new configuration object is constructed with defaults and returned.
    pub fn load_config(alt_config_filepath: Option<&String>) -> ConfigurationFile {
        let filepath: Option<PathBuf> = locate_config_file("config.yaml", alt_config_filepath);

        // if we found a file, deserialize it as yaml
        if let Some(found_file) = filepath {
            match std::fs::read_to_string(&found_file) {
                Ok(plain_string) => {
                    match serde_yaml::from_str::<ConfigurationFile>(plain_string.as_str()) {
                        Ok(cfg) => {
                            return cfg;
                        }
                        Err(err) => {
                            log::error!(
                                "Failed to deserialize the configuration file ({:?}): {}",
                                found_file,
                                err
                            );
                        }
                    };
                }
                Err(err) => log::error!(
                    "Failed to load the configuration file ({:?}): {}",
                    found_file,
                    err
                ),
            };
        }

        // if we made it here, no config file was found, or if it was found, it could not be deserialized as yaml.
        log::warn!(
            "Using a default configuration file from memory since none were located to be read."
        );
        return Default::default();
    }

    // This function takes in a string that should match a conifgured model or filepath and returns
    // the matching model configuration object.
    pub fn find_model_configuration(&self, name_or_path: &str) -> Option<ConfiguredLlm> {
        for m in &self.models {
            if let Some(local_path) = &m.path {
                if m.name.eq_ignore_ascii_case(name_or_path)
                    || local_path.eq_ignore_ascii_case(name_or_path)
                {
                    return Some(m.clone());
                }
            } else {
                if m.name.eq_ignore_ascii_case(name_or_path) {
                    return Some(m.clone());
                }
            }
        }

        None
    }
}

// loads a configuration file in the following order:
//  1) alternate path provided as parameter
//  2) 'platform' config folder (e.g. /home/alice/.config or C:\Users\Alice\AppData\Roaming or /Users/Alice/Library/Application Support)
//  3) next to the binary in the working folder
// if those fail to load, then a new configuration object is constructed with defaults and returned.
pub fn locate_config_file(filename: &str, alt_path: Option<&String>) -> Option<PathBuf> {
    let mut filepath: Option<PathBuf> = None;

    // specified alternate config file
    if let Some(alt) = alt_path {
        let p = Path::new(alt.as_str());
        if p.exists() {
            filepath = Some(p.to_path_buf());
        }
    }

    // try the 'platform' config file location
    if filepath.is_none() {
        if let Some(base_dirs) = BaseDirs::new() {
            let p = Path::new(&base_dirs.config_dir())
                .join(APPLICATION_CONFIG_FOLDER_NAME)
                .join(filename);
            if p.exists() {
                filepath = Some(p);
            }
        }
    }

    // last attempt, look parallel next to the executable
    if filepath.is_none() {
        let p = Path::new(filename);
        if p.exists() {
            filepath = Some(p.to_path_buf());
        }
    }

    filepath
}

// returns the folder path for a given character.
// note: this currently returns `characters/{name}-logs/`
pub fn get_log_folder(char_name: &str) -> std::path::PathBuf {
    let log_path = std::path::Path::new("characters").join(format!("{}-logs", char_name));

    return log_path;
}
