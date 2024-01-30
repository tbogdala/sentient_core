use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
};

use anyhow::{anyhow, Context, Result};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

#[cfg(feature = "sentence_similarity")]
use candle_core::Tensor;

// at no point should this be actually used as a 'Tensor' type
// ... it's only meant to satisfy the compiler when embeddings
// will not be used due to configuration.
#[cfg(not(feature = "sentence_similarity"))]
type Tensor = u8;

use crate::{config::CharacterFileYaml, memories::MemoryFile};

const CURRENT_CHATLOG_VERSION: u32 = 1;
static DEFAULT_ENTITY_NAME: &str = "Unknown";

// this is one turn of a conversation in the chat log (e.g. the AI's response or the human's query).
// at present all embeddings generated for the ChatLogItem are kept without regard to which *parts*
// of the `lines` each embedding covers, though you can reverse engineer that if you know the token
// cutoff for the embedding model.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChatLogItem {
    // meant to identify the 'speaker' for this chatlog item.
    pub entity: String,

    // the lines contained in the message
    pub lines: Vec<String>,

    #[serde(skip)]
    pub embeddings: Vec<Tensor>,
}
// customize partialeq to only care about the serializable data
impl PartialEq for ChatLogItem {
    fn eq(&self, other: &Self) -> bool {
        self.lines == other.lines
    }
}
impl ChatLogItem {
    // creates a new ChatLogItem with empty content.=
    pub fn new() -> Self {
        Self {
            entity: DEFAULT_ENTITY_NAME.to_owned(),
            lines: Vec::new(),
            embeddings: Vec::new(),
        }
    }

    // creates a new ChatLogItem using the vector of Strings passed in
    #[allow(dead_code)]
    pub fn new_from_strings(entity: String, v: &Vec<String>) -> Self {
        Self {
            entity,
            lines: v.to_owned(),
            embeddings: Vec::new(),
        }
    }

    // creates a new ChatLogItem using the String passed in and automatically
    // splits it into lines based on newline characters.
    pub fn new_from_str(entity: String, s: &str) -> Self {
        let mut new_item = ChatLogItem::new();
        new_item.entity = entity;
        for line in s.lines() {
            new_item.lines.push(line.to_owned());
        }
        new_item
    }

    // adds the string to the last line in the log item, breaking apart any
    // additional new lines in the incoming string.
    // if the log item is empty, then it is made the only string.
    pub fn add_to_last(&mut self, s: &str) {
        match self.lines.pop() {
            Some(mut last) => {
                last.push_str(s);
                for line in last.lines() {
                    self.lines.push(line.to_owned());
                }
            }
            None => self.lines.push(s.to_owned()),
        }
    }

    // returns a new string that is the concatenation of all the log item strings
    // and prepended with the entity name.
    pub fn get_name_and_items_as_string(&self) -> String {
        [self.entity.clone(), ": ".to_owned(), self.lines.join("\n")].concat()
    }

    // returns a new string that is the concatenation of all the log item strings
    // and prepended with the entity name.
    pub fn get_items_as_string(&self) -> String {
        self.lines.join("\n")
    }

    pub fn replace_items_with_string(&mut self, paragraph: String) {
        self.lines.clear();
        if !paragraph.is_empty() {
            let splits: Vec<&str> = paragraph.split("\n").collect();
            for s in splits {
                self.lines.push(s.to_string());
            }
        }
    }
}

// this struct denotes other participants in the log, though none of them
// are the primary owner.
#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct Participant {
    // an optional model configuration name to load a different model for text
    // inference instead of the original model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_config_name: Option<String>,

    // the relative filepath for the character yaml file.
    pub character_filepath: String,
}

// this is an opaque struct for managing the chatlog for the chat ui,
// and the primary goal should be clean API and hiding implementation details
#[derive(Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ChatLog {
    // last used filepath to load log, if applicable - not updated on saves
    #[serde(skip)]
    last_used_filepath: Option<PathBuf>,

    // the version counter for the log file - should be changed in the
    // app upon breaking changes.
    version: u32,

    // if supplied, can be passed into the prompt templates for the model under
    // the <|user_description|> tag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_description: Option<String>,

    // if supplied, this defines other characters that can be brought into
    // the chat, when placed into multi-chat mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub other_participants: Option<Vec<Participant>>,

    // vector of relative paths for memory files to load for this chatlog
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_files: Option<Vec<String>>,

    // the loaded memory key/value pairs coming from all the files listed in memory_files
    #[serde(skip)]
    pub loaded_memory: HashMap<String, Vec<String>>,

    // the context description for this log file, and is used in prompt temlates
    // under the <|current_context|> tag.
    pub current_context: String,

    // the main content of the chatlog; all of the items in the conversation
    items: Vec<ChatLogItem>,
}
impl ChatLog {
    #[allow(dead_code)]
    // Creates a new instance of the ChatLog with an empty vector of items
    pub fn new() -> Self {
        let items = Vec::new();

        Self {
            items,
            version: CURRENT_CHATLOG_VERSION,
            current_context: String::new(),
            other_participants: None,
            user_description: None,
            memory_files: None,
            loaded_memory: HashMap::new(),
            last_used_filepath: None,
        }
    }

    // creates a new ChatLog from a plaintext log file while recognizing the passed in names.
    // names should include the whole tag you want to match (e.g. "John:" or "Jane:").
    // NOTE: there are assumptions made about the log:
    //   * Each log line should start with a name, followed by a colon, indicating the name of the speaker/actor
    //   * For multiline responses, every line that doesn't start with a name colon gets
    //     attached to the line above it.
    #[allow(dead_code)]
    pub fn new_from_text_file(fp: &PathBuf, names: Vec<String>) -> Result<ChatLog> {
        let f = File::open(&fp).context("Failed to open the file.")?;
        let reader = BufReader::new(f);
        let mut chatlog = ChatLog::new();

        let mut name_buffer = String::new();
        let mut line_buffer = Vec::new();
        for line_res in reader.lines() {
            if let Ok(line) = line_res {
                let mut matched_name = String::new();
                for name in &names {
                    if line.starts_with(name) {
                        if name.ends_with(":") {
                            let mut trimmed_name = name.to_owned();
                            trimmed_name.pop();
                            matched_name = trimmed_name;
                        } else {
                            matched_name = name.to_owned();
                        }
                        break;
                    }
                }
                let new_start = !matched_name.is_empty();

                // if we're not starting a line with name tags, just add it to the
                // last message buffer
                if new_start == false {
                    line_buffer.push(line);
                }
                // if we detect a name at the start of the log, then build out
                // the item for the buffered lines and start a new buffer with this msg
                else {
                    // it's possible the line_buffer is empty still if this is the first
                    // message it's getting to...
                    if line_buffer.is_empty() == false {
                        let new_item =
                            ChatLogItem::new_from_strings(name_buffer.to_owned(), &line_buffer);
                        chatlog.items.push(new_item);
                        line_buffer.clear();
                    }
                    name_buffer = matched_name.to_string();
                    line_buffer.push(line);
                }
            }
        }

        // if we have a buffer that isn't empty, then add it to the end of the items
        if line_buffer.is_empty() == false {
            let last_item = ChatLogItem::new_from_strings(name_buffer.to_owned(), &line_buffer);
            chatlog.items.push(last_item);
            line_buffer.clear();
        }

        // update the last used filepath
        chatlog.last_used_filepath = Some(fp.to_owned());

        Ok(chatlog)
    }

    // creates a new chatlog based on the greeting of the character file.
    pub fn new_with_greeting(character_file: &CharacterFileYaml, user_name: &str) -> Self {
        let items = character_file
            .get_greeting(user_name)
            .iter()
            .map(|s| {
                // use this to pull out the first name mentioned in a log entry
                static TALKER_NAME_REGEX: Lazy<Regex> = Lazy::new(|| {
                    Regex::new(r"^([\w\-]+):")
                        .context("Compiling chat UI talker regex.")
                        .unwrap()
                });

                match TALKER_NAME_REGEX.captures(s) {
                    Some(talker_match) => {
                        let detected_name = &talker_match[1];
                        let mut s_copy = s.clone();
                        let named_removed = s_copy.split_off(detected_name.len() + 2);
                        ChatLogItem::new_from_str(detected_name.to_owned(), named_removed.as_str())
                    }
                    None => ChatLogItem::new_from_str(DEFAULT_ENTITY_NAME.to_owned(), s),
                }
            })
            .collect();

        Self {
            items,
            version: CURRENT_CHATLOG_VERSION,
            current_context: character_file.context.to_owned(),
            other_participants: None,
            user_description: None,
            memory_files: None,
            loaded_memory: HashMap::new(),
            last_used_filepath: None,
        }
    }

    // creates a new chatlog based on a deseralized json file
    pub fn new_from_json(fp: &PathBuf) -> Result<Self> {
        let f = File::open(fp).context("Attempting to open json chatlog file")?;
        let bf = BufReader::new(f);
        let mut chatlog: ChatLog =
            serde_json::from_reader(bf).context("Attempting to deserialize chatlog json")?;

        // update the last used filepath
        chatlog.last_used_filepath = Some(fp.to_owned());

        // now try to load any additional memory files
        if let Some(memory_files) = &chatlog.memory_files {
            for memory_file in memory_files {
                let memory_fp = fp.with_file_name(memory_file);
                let memory_file = MemoryFile::load_from_file(&memory_fp)?;
                // for each memory, add it into the loaded memory hashmap
                for memory in &memory_file.memories {
                    let mem_value = chatlog.loaded_memory.entry(memory.key.clone()).or_default();
                    mem_value.push(memory.value.clone());
                }
            }
        }

        Ok(chatlog)
    }

    pub fn save_to_last_used_json_file(&self) -> Result<()> {
        if let Some(fp) = &self.last_used_filepath {
            let json = serde_json::to_string_pretty(self)
                .context("Attempting to serialize the chatlog to json")?;
            std::fs::write(fp, json).context("Attempting to write the chatlog json file")?;

            Ok(())
        } else {
            Err(anyhow!("Last used filepath for the json chatlog is not set, so it cannot be saved with this function call."))
        }
    }

    // saves the chatlog to json text representation and writes it to a file
    pub fn save_to_json_file(&mut self, fp: &PathBuf) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .context("Attempting to serialize the chatlog to json")?;
        std::fs::write(fp, json).context("Attempting to write the chatlog json file")?;

        // update the last used filepath
        self.last_used_filepath = Some(fp.to_owned());

        Ok(())
    }

    // exports the chatlog as a jsonl dataset of input-output pairs with the output
    // being the chatlogitems where entity is a match with the parameter.
    //
    // Note: things may get trickier if the log isn't always a 1:1 format.
    // The algorithm will try to combine all previous messages before the bot,
    // but will only do so as far as the entity names are the same.
    // Basically, multi-chat mode makes things tricker and currently, a decision
    // was made to only include one previous entity in the 'input' field to avoid
    // possible confusion in training.
    pub fn export_dataset_input_ouptut(&self, fp: &PathBuf, entity: &str) -> Result<()> {
        let mut dataset: Vec<InputOutputDatasetItem> = vec![];

        // holds all the previous chatlogitem objects since the last dataset
        // export; will be used as the input once an item from a matching entity is found.
        let mut previous_logitems: Vec<&ChatLogItem> = vec![];

        for cli in self.iter() {
            if cli.entity.eq(entity) {
                if previous_logitems.is_empty() == false {
                    let last_entity = &previous_logitems.last().unwrap().entity;

                    // only use one previous items where the entity's match,
                    // instead of combining all previous items incase of mult-chat
                    let filtered_prev_items: Vec<&ChatLogItem> = previous_logitems
                        .iter()
                        .rev()
                        .take_while(|item| item.entity.eq(last_entity))
                        .cloned()
                        .collect();

                    let joined_input = filtered_prev_items
                        .iter()
                        .map(|item| item.get_items_as_string())
                        .collect::<Vec<String>>()
                        .join("\n");
                    dataset.push(InputOutputDatasetItem {
                        input: joined_input,
                        output: cli.get_items_as_string(),
                    });
                    previous_logitems.clear();
                } else {
                    // so we have a match on the entity but the previous item buffer
                    // is empty. attempt to tack this message onto the end of the last
                    // dataset item's output
                    if let Some(last_item) = dataset.last() {
                        let mut new_item = last_item.clone();
                        new_item.output.push_str("\n");
                        new_item.output.push_str(cli.get_items_as_string().as_str());
                    }
                }
            } else {
                previous_logitems.push(cli);
            }
        }

        let out_file = File::create(fp).context("Attempting to create file for dataset export")?;
        let mut writer = BufWriter::new(out_file);
        for item in dataset {
            let json_string = serde_json::to_string(&item)
                .context("Attempting to serialize dataset item for input-ouput export")?;
            writer
                .write_all(json_string.as_bytes())
                .context("Attempting to write out JSONL row for dataset export.")?;
            writer
                .write_all(b"\n")
                .context("Attempting to write newline to separate JSON items in dataset export.")?;
        }
        writer
            .flush()
            .context("Attempting to flush dataset export buffer.")?;
        Ok(())
    }

    // returns a reference to the ChatLogItem at the specified index
    pub fn get(&self, index: usize) -> Option<&ChatLogItem> {
        self.items.get(index)
    }

    // returns a mutable reference to the ChatLogItem at the specified index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut ChatLogItem> {
        self.items.get_mut(index)
    }

    // returns a reference to the last use PathBuf when loading the log.
    // potentially unset if the log hasn't been saved ever and created from scratch.
    pub fn get_last_used_filepath(&self) -> Option<&PathBuf> {
        self.last_used_filepath.as_ref()
    }

    // returns an iterator over the ChatLogItems in the log
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &ChatLogItem> {
        self.items.iter()
    }

    // returns a reference to the last log item if it exists
    pub fn last(&self) -> Option<&ChatLogItem> {
        self.items.last()
    }

    // returns the count of ChatLogItem objects in the log
    pub fn len(&self) -> usize {
        self.items.len()
    }

    // adds a new ChatLogItem to the end of the log
    pub fn push(&mut self, item: ChatLogItem) {
        self.items.push(item);
    }

    // removes the last item from the log and returns it.
    // will return None if the log is empty.
    pub fn pop(&mut self) -> Option<ChatLogItem> {
        self.items.pop()
    }

    // removes the ChatLogItem at the index and returns it.
    pub fn remove(&mut self, index: usize) -> Option<ChatLogItem> {
        if index < self.items.len() {
            Some(self.items.remove(index))
        } else {
            None
        }
    }
}

#[derive(Serialize, Clone)]
struct InputOutputDatasetItem {
    input: String,
    output: String,
}
