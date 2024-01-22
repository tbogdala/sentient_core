use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{fs::File, io::BufReader, path::PathBuf};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MemoryFile {
    pub memories: Vec<Memory>,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone, Default)]
pub struct Memory {
    pub key: String,
    pub value: String,
}

impl MemoryFile {
    // loads the memory file from a json file
    pub fn load_from_file(fp: &PathBuf) -> Result<Self> {
        let f = File::open(fp).context("Attempting to open json memory file")?;
        let bf = BufReader::new(f);
        let memory_file: MemoryFile =
            serde_json::from_reader(bf).context("Attempting to deserialize memory json")?;

        Ok(memory_file)
    }

    // saves the memory file out to the file specified
    pub fn save_to_file(&self, fp: &PathBuf) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .context("Attempting to serialize the memory file to json")?;
        std::fs::write(fp, json).context("Attempting to write the memory file json")?;

        Ok(())
    }
}
