use crossterm::event::KeyCode;
use ratatui::{
    prelude::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style, Stylize},
    text::Line,
    widgets::{List, ListItem, Paragraph},
    Frame,
};
use std::path::{Path, PathBuf};

use crate::{
    config::CharacterFileYaml,
    tui::{
        MessageBoxModalWidget, ProcessInputResult, StatefulList, TerminalEvent, TerminalRenderable,
    },
};

const CHARACTERS_FOLDER_NAME: &str = "characters";

pub struct CharacterSelectState {
    character_names: Vec<(String, PathBuf)>,

    // stores the state of the list item to select the log to load
    list_state: StatefulList<String>,

    // contains a modal dialog widget used to show a message or alert to the user
    modal_messagebox: Option<MessageBoxModalWidget>,
}
impl TerminalRenderable for CharacterSelectState {
    fn process_input(&mut self, event: TerminalEvent) -> ProcessInputResult {
        if let Some(modal) = self.modal_messagebox.as_mut() {
            modal.process_input(event);
            if modal.is_finished {
                self.modal_messagebox = None;
            }
        } else if let TerminalEvent::Key(key) = event {
            if key.code == KeyCode::Esc {
                return ProcessInputResult::ChangeScene(
                    crate::application::ApplicationState::MainMenu,
                );
            } else if key.code == KeyCode::Char('k') || key.code == KeyCode::Up {
                self.list_state.previous()
            } else if key.code == KeyCode::Char('j') || key.code == KeyCode::Down {
                self.list_state.next()
            } else if key.code == KeyCode::Char('?') {
                let help_strings = "j or down-arrow  = move down\n\
                                    k or up-arrow    = move up\n\
                                    enter            = load selected character\n\
                                    esc              = go back to main menu\n";

                // show the dialog to create a new log
                let modal = MessageBoxModalWidget::new("Command Reference:", help_strings, 60, 60);
                self.modal_messagebox = Some(modal);
            } else if key.code == KeyCode::Enter {
                if let Some(sel_index) = self.list_state.state.selected() {
                    let char_filepath = &self.character_names[sel_index].1;

                    // try to load the yaml for the character
                    let character = CharacterFileYaml::load_character(char_filepath);

                    // default to using the first set of configured parameters
                    return ProcessInputResult::ChangeScene(
                        crate::application::ApplicationState::CharacterLogSelect(character),
                    );
                }
            }
        }

        ProcessInputResult::None
    }

    fn render(&mut self, frame: &mut Frame) {
        let divider = "----------------";
        let divider_len = divider.len();

        // FIXME: This is a poor why of doing it and will only support up to ten characters.
        // Build the menu lines up based on the characters we've scanned at the start
        // of the state switch.
        let menu_lines = vec![Line::from("Character Select".bold()), Line::from(divider)];

        let items: Vec<ListItem> = self
            .character_names
            .iter()
            .map(|(c, _)| {
                let lines = vec![Line::from(c.as_str())];
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

        let vchunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(
                [
                    Constraint::Percentage(20),
                    Constraint::Max(2),
                    Constraint::Min(4),
                ]
                .as_ref(),
            )
            .split(hchunks[1]);

        let title = Paragraph::new(menu_lines).alignment(Alignment::Center);
        frame.render_widget(title, vchunks[1]);

        // now render the character list
        frame.render_stateful_widget(items, vchunks[2], &mut self.list_state.state);

        // Now render any modal boxes over the chat log, only selecting one of them to draw.
        // This *should* mimic the same order that input processing gets called so that
        // there's no confusion.

        if let Some(modal) = &self.modal_messagebox {
            modal.render(frame);
        }
    }
}
impl CharacterSelectState {
    pub fn new() -> Self {
        let mut character_names: Vec<(String, PathBuf)> = Vec::new();
        let mut list_items = vec![];

        // browse the characters folder and pull out all
        // character yaml files.
        let characters_dir_path = Path::new(CHARACTERS_FOLDER_NAME);
        for entry in characters_dir_path.read_dir().unwrap() {
            if let Ok(entry) = entry {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_file() {
                        let fp = entry.path();
                        if let Some(file_ext) = fp.extension() {
                            if file_ext.eq_ignore_ascii_case("yaml") {
                                let filename_root = fp.file_stem().unwrap();
                                let filename_str = filename_root.to_str().unwrap().to_string();
                                list_items.push(filename_str.clone());
                                character_names.push((filename_str, fp))
                            }
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
            character_names,
            list_state,
            modal_messagebox: None,
        }
    }
}
