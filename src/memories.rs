use std::{io::BufReader, fs::File, path::PathBuf};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

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
    pub fn load_from_file(fp: &PathBuf) -> Result<Self> {
        let f = File::open(fp).context("Attempting to open json memory file")?;
        let bf = BufReader::new(f);
        let memory_file: MemoryFile =
            serde_json::from_reader(bf).context("Attempting to deserialize memory json")?;

        Ok(memory_file)
    }
}