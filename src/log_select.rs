use std::{
    fs::DirBuilder,
    path::{Path, PathBuf},
};

use anyhow::Context;
use crossterm::event::{KeyCode, KeyModifiers};
use ratatui::{
    prelude::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style, Stylize},
    text::Line,
    widgets::{List, ListItem, Paragraph},
};

use crate::{
    chatlog::ChatLog,
    config::{get_log_folder, CharacterFileYaml, ConfigurationFile, LOG_FILE_NAME},
    memories::{Memory, MemoryFile},
    tui::{
        Frame, MessageBoxModalWidget, ProcessInputResult, StatefulList, TerminalEvent,
        TerminalRenderable, TextEditingBlockModalWidget,
    },
};

enum LogSelectEditorState {
    NewLogFilename,
    DupeLogFilename,
    ExportDatasetFilename,
}

pub struct LogSelectState {
    // a copy of the configuration loaded for the applciation
    config: ConfigurationFile,

    // the character to locate logs for
    character: CharacterFileYaml,

    // Log files detected in a tuple representing: (log folder, log file)
    logs_found: Vec<(PathBuf, PathBuf)>,

    // stores the state of the list item to select the log to load
    list_state: StatefulList<String>,

    // contains the modal dialog widget used to prompt the user for a variety of tasks
    // and the enum value indicating what is being edited
    log_basic_editor: Option<(LogSelectEditorState, TextEditingBlockModalWidget)>,

    // contains a modal dialog widget used to show a message or alert to the user
    modal_messagebox: Option<MessageBoxModalWidget>,
}
impl TerminalRenderable for LogSelectState {
    fn process_input(&mut self, event: TerminalEvent) -> ProcessInputResult {
        if let Some(modal) = self.modal_messagebox.as_mut() {
            modal.process_input(event);
            if modal.is_finished {
                self.modal_messagebox = None;
            }
        } else if let Some((editor_type, editor)) = self.log_basic_editor.as_mut() {
            editor.process_input(event);
            if editor.is_finished {
                if editor.is_success {
                    match editor_type {
                        LogSelectEditorState::ExportDatasetFilename => {
                            let export_filename = editor.text.to_owned();
                            if let Some(sel_index) = self.list_state.state.selected() {
                                let log_file = &self.logs_found[sel_index].1;
                                let chatlog_res = ChatLog::new_from_json(&log_file);
                                let export_filepath = log_file.with_file_name(export_filename);
                                match chatlog_res {
                                    Ok(chatlog) => {
                                        let res = chatlog.export_dataset_input_ouptut(
                                            &export_filepath,
                                            &self.character.name,
                                        );
                                        if let Err(e) = res {
                                            log::error!(
                                                "Failed to export the chatlog ({:?}): {}",
                                                log_file,
                                                e
                                            )
                                        }
                                    }
                                    Err(err) => {
                                        log::error!(
                                            "Failed to load the chatlog ({:?}): {}",
                                            log_file,
                                            err
                                        )
                                    }
                                };
                            }
                        }

                        LogSelectEditorState::NewLogFilename => {
                            // create the new log
                            let newlog_name = editor.text.to_owned();
                            let log_folder_path = get_log_folder(self.character.name.as_str());
                            let new_log_folder_path = log_folder_path.join(newlog_name);
                            let new_log_file_path = new_log_folder_path.join(LOG_FILE_NAME);
                            if new_log_file_path.exists() {
                                log::error!(
                                        "Log file already exists, so a new one will not be created here: {:?}",
                                        new_log_file_path
                                    );
                            } else {
                                let made_dir = DirBuilder::new()
                                    .recursive(true)
                                    .create(&new_log_folder_path)
                                    .context(
                                        "Attempting to create the directory for the new chatlog",
                                    );
                                if made_dir.is_ok() {
                                    let mut new_log = ChatLog::new_with_greeting(
                                        &self.character,
                                        &self.config.display_name,
                                    );

                                    // generate a new log with a blank memory file automatically
                                    let memory_filename = "memories.json".to_string();
                                    new_log.memory_files = Some(vec![memory_filename.clone()]);
                                    let mut memory_data = MemoryFile::default();
                                    memory_data.memories.push(Memory{
                                        key: "This is a sample memory key - replace with the phrase to search for".into(),
                                        value: "This is a sample memory text that will get included in the <|memory_matches|> template when 'key' is found.".into()
                                    });
                                    let default_memory_file =
                                        new_log_folder_path.join(memory_filename);
                                    memory_data.save_to_file(&default_memory_file)
                                        .context("Attempting to create a default memory file for the character")
                                        .unwrap();

                                    if let Err(err) = new_log.save_to_json_file(&new_log_file_path)
                                    {
                                        log::error!(
                                            "Failed to save the new log file to {:?}: {}",
                                            new_log_file_path,
                                            err
                                        );
                                    } else {
                                        return ProcessInputResult::ChangeScene(
                                            crate::application::ApplicationState::Chat(
                                                self.character.to_owned(),
                                                new_log,
                                            ),
                                        );
                                    }
                                }
                            }
                        }

                        LogSelectEditorState::DupeLogFilename => {
                            if let Some(sel_index) = self.list_state.state.selected() {
                                let source_log_dir = &self.logs_found[sel_index]
                                    .0
                                    .file_name()
                                    .context("Attempting to get the source dir name to duplicate.")
                                    .unwrap();
                                let new_log_dir = editor.text.to_owned();

                                let log_folder_path = get_log_folder(self.character.name.as_str());
                                let src_log_folder_path = log_folder_path.join(source_log_dir);
                                let dst_log_folder_path = log_folder_path.join(new_log_dir);

                                if let Err(err) = copy_files_in_dir(
                                    src_log_folder_path.as_path(),
                                    dst_log_folder_path.as_path(),
                                ) {
                                    log::error!(
                                        "Failed to copy the log folder from {} to {}: {}",
                                        src_log_folder_path.to_str().unwrap_or("<Unknown>"),
                                        dst_log_folder_path.to_str().unwrap_or("<Unknown>"),
                                        err
                                    );
                                } else {
                                    // update the user interface by creating a new instance of
                                    // it and then ripping out the directories found and the list state
                                    let new_lss = LogSelectState::new(
                                        self.character.clone(),
                                        self.config.clone(),
                                    );
                                    self.list_state = new_lss.list_state;
                                    self.logs_found = new_lss.logs_found;
                                }
                            }
                        }
                    }
                }
                self.log_basic_editor = None;
            }
        } else {
            if let TerminalEvent::Key(key) = event {
                if key.code == KeyCode::Esc {
                    return ProcessInputResult::ChangeScene(
                        crate::application::ApplicationState::CharacterSelect,
                    );
                } else if key.code == KeyCode::Char('k') || key.code == KeyCode::Up {
                    self.list_state.previous()
                } else if key.code == KeyCode::Char('j') || key.code == KeyCode::Down {
                    self.list_state.next()
                } else if key.code == KeyCode::Enter {
                    // load the chatlog up and pass it to the chat interface
                    if let Some(sel_index) = self.list_state.state.selected() {
                        let log_file = &self.logs_found[sel_index].1;
                        let chatlog_res = ChatLog::new_from_json(&log_file);
                        match chatlog_res {
                            Ok(chatlog) => {
                                return ProcessInputResult::ChangeScene(
                                    crate::application::ApplicationState::Chat(
                                        self.character.to_owned(),
                                        chatlog,
                                    ),
                                )
                            }
                            Err(err) => {
                                log::error!("Failed to load the chatlog ({:?}): {}", log_file, err)
                            }
                        };
                    }
                } else if key.code == KeyCode::Char('n') {
                    if key.modifiers.contains(KeyModifiers::CONTROL) {
                        // show the dialog to create a new log
                        let ce = TextEditingBlockModalWidget::new(
                            "Enter a name for the new chatlog:".to_owned(),
                            String::new(),
                        );
                        self.log_basic_editor = Some((LogSelectEditorState::NewLogFilename, ce));
                    }
                } else if key.code == KeyCode::Char('o') {
                    if key.modifiers.contains(KeyModifiers::CONTROL) {
                        // show the dialog to create a new exported dataset
                        let ce = TextEditingBlockModalWidget::new(
                            "Enter a name for the exported chatlog dataset:".to_owned(),
                            String::new(),
                        );
                        self.log_basic_editor =
                            Some((LogSelectEditorState::ExportDatasetFilename, ce));
                    }
                } else if key.code == KeyCode::Char('d') {
                    if key.modifiers.contains(KeyModifiers::CONTROL) {
                        let starting_value = if let Some(sel_index) =
                            self.list_state.state.selected()
                        {
                            self.logs_found[sel_index]
                                    .0
                                    .file_name()
                                    .context("Attempting to get directory name of a path for log duplication")
                                    .unwrap()
                                    .to_str()
                                    .context("Converting log filename to string")
                                    .unwrap()
                                    .to_string()
                        } else {
                            String::new()
                        };

                        // show the dialog to duplicate the selected log file
                        let ce = TextEditingBlockModalWidget::new(
                            "Enter a new name for the duplicate chatlog:".to_owned(),
                            starting_value,
                        );
                        self.log_basic_editor = Some((LogSelectEditorState::DupeLogFilename, ce));
                    }
                } else if key.code == KeyCode::Char('?') {
                    let help_strings = "j or down-arrow  = move down\n\
                                        k or up-arrow    = move up\n\
                                        enter            = load selected chatlog\n\
                                        esc              = go back to character select\n\
                                        ctrl-n           = create a new chatlog\n\
                                        ctrl-d           = duplicate existing chatlog with a new name\n\
                                        ctrl-o           = export selected chatlog as a training dataset\n";

                    // show the dialog to create a new log
                    let modal =
                        MessageBoxModalWidget::new("Command Reference:", help_strings, 60, 60);
                    self.modal_messagebox = Some(modal);
                }
            }
        }

        ProcessInputResult::None
    }

    fn render(&mut self, frame: &mut Frame) {
        let divider = "------------";
        let divider_len = divider.len();
        let menu_lines = vec![Line::from("Select a Log".bold()), Line::from(divider)];

        let items: Vec<ListItem> = self
            .logs_found
            .iter()
            .map(|(d, _)| {
                let dir_name = d
                    .file_name()
                    .context("Accessing log directory file_name.")
                    .unwrap()
                    .to_str()
                    .context("Converting log directory name to a string.")
                    .unwrap();
                let lines = vec![Line::from(dir_name)];
                ListItem::new(lines).style(Style::default())
            })
            .collect();

        // start with the divider length as the max width and adjust for any list items
        let mut max_width = divider_len;
        if !items.is_empty() {
            max_width = items
                .iter()
                .max_by(|x, y| x.width().cmp(&y.width()))
                .unwrap()
                .width();
        }

        // TODO: allow customization of 'highlight color'
        let items = List::new(items)
            .highlight_style(
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol(">> ");

        // break things up horizontally to create some padding
        let middle_column_size = 3 + max_width.max(divider_len) as u16;
        let padding_size = (frame.size().width - middle_column_size) / 2;
        let hchunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(
                [
                    Constraint::Max(padding_size),
                    Constraint::Min(middle_column_size),
                    Constraint::Max(padding_size),
                ]
                .as_ref(),
            )
            .split(frame.size());

        // now break things up vertically
        let vchunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(
                [
                    Constraint::Percentage(20),
                    Constraint::Min(2),
                    Constraint::Min(4),
                ]
                .as_ref(),
            )
            .split(hchunks[1]);

        // render the header
        let title = Paragraph::new(menu_lines).alignment(Alignment::Center);
        frame.render_widget(title, vchunks[1]);

        // now render the log list
        frame.render_stateful_widget(items, vchunks[2], &mut self.list_state.state);

        // Now render any modal boxes over the chat log, only selecting one of them to draw.
        // This *should* mimic the same order that input processing gets called so that
        // there's no confusion.

        if let Some(modal) = &self.modal_messagebox {
            modal.render(frame);
        }
        // user is attempting to create a new chatlog?
        else if let Some((_, editor)) = &self.log_basic_editor {
            editor.render(frame);
        }
    }
}
impl LogSelectState {
    pub fn new(character: CharacterFileYaml, config: ConfigurationFile) -> Self {
        // build a list of potential log files
        let mut logs_found: Vec<(PathBuf, PathBuf)> = Vec::new();
        let mut list_items = vec![];
        let log_folder = get_log_folder(character.name.as_str());

        // if this is a new character, the log folder might not exist.
        // create a new one and put a default chatlog in there.
        if !log_folder.exists() {
            let default_log_dir = log_folder.join("default");
            let default_log_file = default_log_dir.join("log.json");
            DirBuilder::new()
                .recursive(true)
                .create(&default_log_dir)
                .unwrap();

            // generate a new log with a blank memory file automatically
            let memory_filename = "memories.json".to_string();
            let mut new_chatlog = ChatLog::new_with_greeting(&character, &config.display_name);
            new_chatlog.memory_files = Some(vec![memory_filename.clone()]);
            new_chatlog
                .save_to_json_file(&default_log_file)
                .context("Attempting to create a default chatlog for the character")
                .unwrap();

            let mut memory_data = MemoryFile::default();
            memory_data.memories.push(Memory{
                key: "This is a sample memory key - replace with the phrase to search for".into(),
                value: "This is a sample memory text that will get included in the <|memory_matches|> template when 'key' is found.".into()
            });
            let default_memory_file = default_log_dir.join(memory_filename);
            memory_data
                .save_to_file(&default_memory_file)
                .context("Attempting to create a default memory file for the character")
                .unwrap();
        }

        for entry in log_folder
            .read_dir()
            .expect("Attempting to read the character log directory to scan for logs failed.")
        {
            if let Ok(entry) = entry {
                if let Ok(file_type) = entry.file_type() {
                    // all directories in the log folder are considered for potentially being a log folder
                    if file_type.is_dir() {
                        let log_folder_path = entry.path();
                        let file_path = log_folder_path.join(crate::config::LOG_FILE_NAME);
                        if file_path.exists() {
                            let dir_name = log_folder_path
                                .file_name()
                                .context("Accessing log directory file_name.")
                                .unwrap()
                                .to_str()
                                .context("Converting log directory name to a string.")
                                .unwrap();
                            list_items.push(dir_name.to_string());
                            logs_found.push((log_folder_path, file_path));
                        }
                    }
                }
            }
        }

        let mut list_state = StatefulList::with_items(list_items);
        if !list_state.items.is_empty() {
            list_state.state.select(Some(0));
        }

        Self {
            config,
            character,
            logs_found,
            list_state,
            log_basic_editor: None,
            modal_messagebox: None,
        }
    }
}

// this function only copies files from one directory to another; directories are skipped.
// the destination directory will be created if it doesn't exist already
fn copy_files_in_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        if let Ok(entry) = entry {
            let path = entry.path();
            if path.is_file() {
                std::fs::copy(
                    &path,
                    dst.join(
                        path.file_name()
                            .context(
                                "Getting the filename for the source file during directory copy.",
                            )
                            .unwrap(),
                    ),
                )?;
            }
        }
    }
    Ok(())
}
