use std::{fs::DirBuilder, path::PathBuf};

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
    tui::{
        Frame, MessageBoxModalWidget, ProcessInputResult, StatefulList, TerminalEvent,
        TerminalRenderable, TextEditingBlockModalWidget,
    },
};

pub struct LogSelectState {
    // a copy of the configuration loaded for the applciation
    config: ConfigurationFile,

    // the character to locate logs for
    character: CharacterFileYaml,

    // Log files detected in a tuple representing: (log folder, log file)
    logs_found: Vec<(PathBuf, PathBuf)>,

    // stores the state of the list item to select the log to load
    list_state: StatefulList<String>,

    // contains the modal dialog widget used to prompt the user to pick a new log name
    newlog_name_editor: Option<TextEditingBlockModalWidget>,

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
        } else if let Some(name_editor) = self.newlog_name_editor.as_mut() {
            name_editor.process_input(event);
            if name_editor.is_finished {
                if name_editor.is_success {
                    // create the new log
                    let newlog_name = name_editor.text.to_owned();
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
                            .create(new_log_folder_path)
                            .context("Attempting to create the directory for the new chatlog");
                        if made_dir.is_ok() {
                            let new_log = ChatLog::new_with_greeting(
                                &self.character,
                                &self.config.display_name,
                            );
                            if let Err(err) = new_log.save_to_json_file(&new_log_file_path) {
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
                self.newlog_name_editor = None;
            }
        } else {
            if let TerminalEvent::Key(key) = event {
                if key.code == KeyCode::Esc {
                    return ProcessInputResult::ChangeScene(
                        crate::application::ApplicationState::CharacterSelect,
                    );
                } else if key.code == KeyCode::Char('k') {
                    self.list_state.previous()
                } else if key.code == KeyCode::Char('j') {
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
                        self.newlog_name_editor = Some(ce);
                    }
                } else if key.code == KeyCode::Char('?') {
                    let help_strings = "j      = move down\n\
                                        k      = move up\n\
                                        enter  = load selected chatlog\n\
                                        esc    = go back to character select\n\
                                        ctrl-n = crate a new chatlog\n";

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

        let max_width = items
            .iter()
            .max_by(|x, y| x.width().cmp(&y.width()))
            .unwrap()
            .width();

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
        // user is attempt to create a new chatlog?
        else if let Some(name_editor) = &self.newlog_name_editor {
            name_editor.render(frame);
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
            let new_chatlog = ChatLog::new_with_greeting(&character, &config.display_name);
            new_chatlog
                .save_to_json_file(&default_log_file)
                .context("Attempting to create a default chatlog for the character")
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
            newlog_name_editor: None,
            modal_messagebox: None,
        }
    }
}
