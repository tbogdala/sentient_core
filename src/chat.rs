use anyhow::Context;
use crossbeam::channel::{Receiver, Sender};
use crossterm::event::{KeyCode, KeyModifiers};
use rand::prelude::*;
use ratatui::prelude::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Chart, Clear, Dataset, Paragraph, Sparkline};
use ratatui::Frame;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use unicode_segmentation::UnicodeSegmentation;

use crate::chatlog::{ChatLog, ChatLogItem};
use crate::config::*;
use crate::llm_engine::{LlmEngineCommand, TextInferenceContext};
use crate::llm_engine::{self, LlmEngineRequest, LlmEngineResponse};
use crate::tui::{
    centered_rect, slice_up_string, MessageBoxModalWidget, ProcessInputResult, TerminalEvent,
    TerminalRenderable, TextEditingBlockModalWidget,
};

// This enum is used to identify how the editor_widget should behave so that only
// one widget is needed, since it's modal anyway.
enum ChatEditorState {
    // used to update the chatlog context
    ChatlogContext,

    // used to update the user's description context
    UserDescription,

    // used to update the chatlog item that is 'current' - as determined by the 'chatlog_scroll` member
    ChatlogItem,

    // used to input a 'command' to sentient_core itself as if from a developer console
    SlashCommand,
}

pub struct ChatState {
    // a copy of the configuration file passed into the UI at creation
    config: ConfigurationFile,

    character: CharacterFileYaml,

    // a tuple of character file and optional model_config_name to load for generation
    other_participants: Vec<(CharacterFileYaml, Option<String>)>,

    chatlog: ChatLog,
    chatlog_scroll: usize,
    current_parameters: ConfiguredParameters,
    manual_reply_mode: bool,

    send_to_server: Sender<LlmEngineRequest>,
    send_cmd_to_server: Sender<LlmEngineCommand>,
    recv_on_client: Receiver<LlmEngineResponse>,

    editing_reply: bool,
    editing_parameters: bool,
    reply_text: String,

    waiting_for_operation: bool,

    // The character that is currently causing the `waiting_for_operation`
    // field to be set to true ... basically, the character who we're waiting on text
    // for. If set to None, that mean's it's the user.
    waiting_for_character: Option<CharacterFileYaml>,

    // The optional string representing the incoming text prediction for the
    // character specified by `waiting_for_character`.
    in_flight_text: Option<String>,

    progress_widget: Option<ProgressBarScopeSignal>,

    // contains a modal dialog widget used to show a message or alert to the user
    modal_messagebox: Option<MessageBoxModalWidget>,

    // a tuple that may be the active editor widget, with the intended behavior
    // being indicated by the ChatEditorState enum.
    editor_widget: Option<(ChatEditorState, TextEditingBlockModalWidget)>,
}
impl ChatState {
    // Creates a new ChatState for the selected character.
    pub fn new(
        character: CharacterFileYaml,
        chatlog: ChatLog,
        inference_parameters: Option<&ConfiguredParameters>,
        config: ConfigurationFile,
        send_to_server: Sender<LlmEngineRequest>,
        send_cmd_to_server: Sender<LlmEngineCommand>,
        recv_on_client: Receiver<LlmEngineResponse>,
    ) -> ChatState {
        let config = config.clone();

        let current_parameters = match inference_parameters {
            Some(params) => ConfiguredParameters {
                name: params.name.clone(),
                top_k: params.top_k,
                top_p: params.top_p,
                min_p: params.min_p,
                temperature: params.temperature,
                frequency_penalty: params.frequency_penalty,
                repeat_penalty: params.repeat_penalty,
                repeat_penalty_range: params.repeat_penalty_range,
                mirostat: params.mirostat,
                mirostat_eta: params.mirostat_eta,
                mirostat_tau: params.mirostat_tau,
            },
            None => ConfiguredParameters::default(),
        };

        let send_to_server = send_to_server.clone();
        let recv_on_client = recv_on_client.clone();

        ChatState {
            config,
            character,
            other_participants: Vec::new(),
            chatlog,
            chatlog_scroll: 0,
            current_parameters,
            manual_reply_mode: false,
            send_to_server,
            send_cmd_to_server,
            recv_on_client,
            editing_reply: false,
            editing_parameters: false,
            reply_text: String::new(),
            waiting_for_operation: false,
            waiting_for_character: None,
            in_flight_text: None,
            progress_widget: None,
            modal_messagebox: None,
            editor_widget: None,
        }
    }

    // saves the file out to the file it was last loaded from and returns a bool
    // indicating if the log was successfully saved. if no last_used_filepath is
    // set, then the function doesn't do anything and returns false.
    fn save_chatlog_to_last_used(&self) -> bool {
        // save the log file out if the last-used filepath was set
        if let Err(err) = self.chatlog.save_to_last_used_json_file() {
            log::error!(
                "Failed to write the chatlog after receiving next text inference response: {}",
                err
            );
            false
        } else {
            true
        }
    }

    fn process_incoming_llm_engine_messages(&mut self) {
        // see if there are any incoming messages from the server
        while self.recv_on_client.is_empty() == false {
            match self.recv_on_client.try_recv() {
                Ok(llm_engine::LlmEngineResponse::NewText(maybe_resp, context)) => {
                    if let Some(resp) = maybe_resp {
                        //TODO: consider a different way of getting vector embeddings back from the thread
                        self.chatlog = context.chatlog;

                        if context.should_continue == false {
                            let new_item = ChatLogItem::new_from_str(
                                context.character.name.to_owned(),
                                resp.trim(),
                            );
                            self.chatlog.push(new_item);
                        } else {
                            // if we don't have a log item to append we just make a new one
                            let mut last_item = self.chatlog.pop().unwrap_or_default();
                            last_item.add_to_last(resp.as_str());
                            self.chatlog.push(last_item);
                        }

                        // save the log file out
                        let _ = self.save_chatlog_to_last_used();

                        // clean up user interface bits now that the prediction is finished
                        self.hide_progress_bar();
                        self.in_flight_text = None;
                    } else {
                        log::error!("Response for the text inferrence was empty.");
                    }
                }

                Ok(llm_engine::LlmEngineResponse::NewTextFragment(token)) => {
                    if self.in_flight_text == None {
                        self.in_flight_text = Some(String::new());
                    }
                    if let Some(in_flight) = self.in_flight_text.as_mut() {
                        in_flight.push_str(token.as_str());
                    }
                }

                _ => {}
            }
        }
    }

    fn process_input_for_editing_parameters(&mut self, event: TerminalEvent) {
        if let TerminalEvent::Key(key) = event {
            match key.code {
                KeyCode::Esc => {
                    self.editing_parameters = false;
                }
                KeyCode::Enter => {
                    self.editing_parameters = false;
                }
                KeyCode::Char('h') | KeyCode::Left => {
                    for (i, pset) in self.config.parameters.iter().enumerate() {
                        if self
                            .current_parameters
                            .name
                            .eq_ignore_ascii_case(pset.name.as_str())
                        {
                            self.current_parameters = if i == 0 {
                                self.config.parameters.last().unwrap().clone()
                            } else {
                                self.config.parameters[i - 1].clone()
                            };
                            break;
                        }
                    }
                }
                KeyCode::Char('l') | KeyCode::Right => {
                    for (i, pset) in self.config.parameters.iter().enumerate() {
                        if self
                            .current_parameters
                            .name
                            .eq_ignore_ascii_case(pset.name.as_str())
                        {
                            self.current_parameters = if i >= self.config.parameters.len() - 1 {
                                self.config.parameters.first().unwrap().clone()
                            } else {
                                self.config.parameters[i + 1].clone()
                            };
                            break;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn process_input_for_editing_replies(&mut self, event: TerminalEvent) {
        if let TerminalEvent::Key(key) = event {
            match key.code {
                KeyCode::Esc => {
                    self.editing_reply = false;
                }
                KeyCode::Backspace => {
                    self.reply_text.pop();
                }
                KeyCode::Char(to_insert) => {
                    self.reply_text.push(to_insert);
                }
                KeyCode::Enter => {
                    let mut trimmed_reply_text = self.reply_text.trim().to_string();

                    // if the reply text is empty, we just ignore all of this and return
                    if trimmed_reply_text.is_empty() {
                        return;
                    }

                    // check to see if the string just ends with a non-escaped "\n" and if so,
                    // just replace that with a newline character.
                    if trimmed_reply_text.ends_with("\\n") {
                        trimmed_reply_text.pop();
                        trimmed_reply_text.pop();
                        trimmed_reply_text.push_str("\n");
                        self.reply_text = trimmed_reply_text;
                        return;
                    }

                    // officially add the message we sent to the log
                    let new_message = ChatLogItem::new_from_str(
                        self.config.display_name.clone(),
                        self.reply_text.as_str(),
                    );
                    self.chatlog.push(new_message);
                    self.reply_text.clear();
                    self.editing_reply = false;

                    // save the log file out
                    let _ = self.save_chatlog_to_last_used();

                    // if we're not in manual reply mode, automatically run inferrence
                    if self.manual_reply_mode == false {
                        let context = TextInferenceContext {
                            character: self.character.clone(),
                            model_config_override: None,
                            chatlog_owner: self.character.clone(),
                            other_participants: self.other_participants.clone(),
                            chatlog: self.chatlog.clone(),
                            should_continue: false,
                            parameters: self.current_parameters.clone(),
                        };

                        let msg = llm_engine::LlmEngineRequest::TextInference(context);
                        if let Err(err) = self.send_to_server.send(msg) {
                            log::error!("Error during text infer: {}", err);
                        }

                        self.show_progress_bar(self.character.clone());
                    }
                }
                _ => {}
            }
        }
    }

    fn process_input_for_viewing_chatlog(&mut self, event: TerminalEvent) -> ProcessInputResult {
        if let TerminalEvent::Key(key) = event {
            if key.code == KeyCode::Esc {
                // test to see if text is getting predicted and if so, cancel that request.
                if self.in_flight_text.is_some() {
                    if let Err(err) = self.send_cmd_to_server.send(LlmEngineCommand::CancelTextInference) {
                        log::error!("Error while attempting to cancel text inference: {}", err);
                    }
                } else {
                    return ProcessInputResult::ChangeScene(
                        crate::application::ApplicationState::MainMenu,
                    );
                }
            } else if key.code == KeyCode::Char('y') {
                if key.modifiers.contains(KeyModifiers::CONTROL) {
                    let context = TextInferenceContext {
                        character: self.character.clone(),
                        model_config_override: None,
                        chatlog_owner: self.character.clone(),
                        other_participants: self.other_participants.clone(),
                        chatlog: self.chatlog.clone(),
                        should_continue: false,
                        parameters: self.current_parameters.clone(),
                    };
                    let msg = llm_engine::LlmEngineRequest::TextInference(context);
                    if let Err(err) = self.send_to_server.send(msg) {
                        log::error!("Error during text infer additional request: {}", err);
                    }
                    self.show_progress_bar(self.character.clone());
                }
            } else if key.code == KeyCode::Char('r') {
                if key.modifiers.contains(KeyModifiers::CONTROL) {
                    let last_message = self.chatlog.pop();
                    if last_message.is_none() {
                        return ProcessInputResult::None; // can't regenerate nothing, not even with AI.
                    }

                    // save the log file out
                    let _ = self.save_chatlog_to_last_used();

                    let mut context = TextInferenceContext {
                        character: self.character.clone(),
                        model_config_override: None,
                        chatlog_owner: self.character.clone(),
                        other_participants: self.other_participants.clone(),
                        chatlog: self.chatlog.clone(),
                        should_continue: false,
                        parameters: self.current_parameters.clone(),
                    };

                    // check to see if the last message was sent by the 'main' character
                    // or one of the other participants
                    if let Some(lastmsg) = last_message {
                        if !lastmsg.entity.eq(self.character.name.as_str()) {
                            if !self.other_participants.is_empty() {
                                // find the first match and update the request context
                                for (character, model_ovrride) in &self.other_participants {
                                    if lastmsg.entity.eq(character.name.as_str()) {
                                        context.character = character.clone();
                                        if let Some(ovrride) = model_ovrride {
                                            context.model_config_override = Some(ovrride.clone());
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    self.show_progress_bar(context.character.clone());

                    let msg = llm_engine::LlmEngineRequest::TextInference(context);
                    if let Err(err) = self.send_to_server.send(msg) {
                        log::error!("Error during text infer redo request: {}", err);
                    }
                } else {
                    // regular 'r' is for reply
                    self.editing_reply = true;
                }
            } else if key.code == KeyCode::Char('t') {
                if key.modifiers.contains(KeyModifiers::CONTROL) {
                    // ctrl + t is for continue
                    let mut context = TextInferenceContext {
                        character: self.character.clone(),
                        model_config_override: None,
                        chatlog_owner: self.character.clone(),
                        other_participants: self.other_participants.clone(),
                        chatlog: self.chatlog.clone(),
                        should_continue: true,
                        parameters: self.current_parameters.clone(),
                    };

                    // check to see if the last message was sent by the 'main' character
                    // or one of the other participants
                    if let Some(lastmsg) = self.chatlog.last() {
                        if !lastmsg.entity.eq(self.character.name.as_str()) {
                            if !self.other_participants.is_empty() {
                                // find the first match and update the request context
                                for (character, model_ovrride) in &self.other_participants {
                                    if lastmsg.entity.eq(character.name.as_str()) {
                                        context.character = character.clone();
                                        if let Some(ovrride) = model_ovrride {
                                            context.model_config_override = Some(ovrride.clone());
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    self.show_progress_bar(context.character.clone());

                    let msg = llm_engine::LlmEngineRequest::TextInference(context);
                    if let Err(err) = self.send_to_server.send(msg) {
                        log::error!("Error during text infer redo request: {}", err);
                    }
                }
            } else if key.code == KeyCode::Char('p') {
                self.editing_parameters = true;
            } else if key.code == KeyCode::Char('j') || key.code == KeyCode::Down {
                if self.config.display_chatlog_downward.unwrap_or_default() {
                    if self.chatlog_scroll > 0 {
                        self.chatlog_scroll -= 1;
                    }
                } else {
                    self.chatlog_scroll =
                        std::cmp::min(self.chatlog_scroll + 1, self.chatlog.len());
                }
            } else if key.code == KeyCode::Char('k') || key.code == KeyCode::Up {
                if self.config.display_chatlog_downward.unwrap_or_default() {
                    self.chatlog_scroll =
                        std::cmp::min(self.chatlog_scroll + 1, self.chatlog.len());
                } else {
                    if self.chatlog_scroll > 0 {
                        self.chatlog_scroll -= 1;
                    }
                }
            } else if key.code == KeyCode::Char('x') {
                if key.modifiers.contains(KeyModifiers::CONTROL) {
                    // ctrl + x for deleting selected entry
                    let index = self.get_currently_select_chatlogitem_index();
                    self.chatlog.remove(index);

                    // save the log file out
                    let _ = self.save_chatlog_to_last_used();
                }
            } else if key.code == KeyCode::Char('o') {
                if key.modifiers.contains(KeyModifiers::CONTROL) {
                    let user_desc = self.chatlog.user_description.clone().unwrap_or_default();
                    let ce =
                        TextEditingBlockModalWidget::new("User Description".to_owned(), user_desc);
                    self.editor_widget = Some((ChatEditorState::UserDescription, ce));
                } else {
                    let ce = TextEditingBlockModalWidget::new(
                        "Conversation Context".to_owned(),
                        self.chatlog.current_context.to_owned(),
                    );
                    self.editor_widget = Some((ChatEditorState::ChatlogContext, ce));
                }
            } else if key.code == KeyCode::Char('e') {
                let index = self.get_currently_select_chatlogitem_index();
                if let Some(cli) = self.chatlog.get(index) {
                    let ce = TextEditingBlockModalWidget::new(
                        "Edit Message".to_owned(),
                        cli.get_items_as_string(),
                    );
                    self.editor_widget = Some((ChatEditorState::ChatlogItem, ce));
                } else {
                    log::error!("Failed to get the chatlog item at index {}", index);
                }
            } else if key.code == KeyCode::Char('m') {
                self.manual_reply_mode = !self.manual_reply_mode;
                if self.manual_reply_mode {
                    let log_path = self
                        .chatlog
                        .get_last_used_filepath()
                        .context("Getting last used chatlog filepath.")
                        .unwrap();
                    if let Some(others) = self.chatlog.other_participants.as_ref() {
                        self.other_participants.clear();
                        for other_participant in others {
                            let another_char = log_path
                                .with_file_name(other_participant.character_filepath.as_str());
                            let character = CharacterFileYaml::load_character(&another_char);
                            self.other_participants
                                .push((character, other_participant.model_config_name.to_owned()));
                        }
                        self.modal_messagebox = Some(MessageBoxModalWidget::new(
                            "Information", 
                            "Multi-chat Mode enabled! Responses will no longer be automatically generated. Use number keys to trigger responses for the matching character.", 60, 30));
                    } else {
                        self.manual_reply_mode = false;
                        self.modal_messagebox = Some(MessageBoxModalWidget::new(
                            "Information", 
                            "Other participants need to be defined in the chatlog in order to enable multi-chat mode.", 60, 30));
                    }
                } else {
                    self.other_participants.clear();
                    self.modal_messagebox = Some(MessageBoxModalWidget::new(
                        "Information", 
                        "Multi-chat Mode disabled! Chat responses will be automatically generated for the main character.", 60, 30));
                }
            } else if key.code == KeyCode::Char('/') {
                let ce =
                    TextEditingBlockModalWidget::new("Execute Command".to_owned(), "".to_string());
                self.editor_widget = Some((ChatEditorState::SlashCommand, ce));
            } else if key.code == KeyCode::Char('?') {
                let help_strings = "j or down-arrow  = scroll chatlog down\n\
                                    k or up-arrow    = scroll chatlog up\n\
                                    r      = type a new message to the AI (esc to cancel)\n\
                                    ctrl-r = regenerate the AI's last response\n\
                                    ctrl-t = continues the AI's last response\n\
                                    ctrl-y = generate another AI response manually\n\
                                    ctrl-x = delete the currently selected chatlog item\n\
                                    o      = set the current context description for the chatlog\n\
                                    ctrl-o = regenerate the AI's last response\n\
                                    e      = edit the currently selected chatlog item\n\
                                    /      = execute an editor command
                                    esc    = exit back to the main menu\n\
                                    \n\
                                    m      = enter multi-chat mode\n\
                                    <1>    = generate a reply for the main AI character\n\
                                    <2-0>  = generate a reply for subesquent 'other participants'\n\
                                    \n\
                                    p      = select a parameter configuration for inference\n\
                                    h or left-arrow   = select parameter config to the left\n\
                                    l or right-arrow  = select parameter config to the right";

                // show the dialog to create a new log
                let modal = MessageBoxModalWidget::new("Command Reference:", help_strings, 60, 60);
                self.modal_messagebox = Some(modal);
            } else if self.manual_reply_mode && key.code == KeyCode::Char('1') {
                let context = TextInferenceContext {
                    character: self.character.clone(),
                    model_config_override: None,
                    chatlog_owner: self.character.clone(),
                    other_participants: self.other_participants.clone(),
                    chatlog: self.chatlog.clone(),
                    should_continue: false,
                    parameters: self.current_parameters.clone(),
                };
                let msg = llm_engine::LlmEngineRequest::TextInference(context);
                if let Err(err) = self.send_to_server.send(msg) {
                    log::error!("Error during text infer additional request: {}", err);
                }
                self.show_progress_bar(self.character.clone());
            } else if self.manual_reply_mode {
                // the case for the normal character is handled above, so this
                // just covers any of the other participants.
                match key.code {
                    KeyCode::Char(c) if c.is_digit(10) => {
                        let mut digit = c.to_digit(10).unwrap();
                        if digit == 0 {
                            digit = 10;
                        }
                        let index = 0.max(digit - 2) as usize;
                        if !self.other_participants.is_empty() {
                            let other_char = self
                                .other_participants
                                .get(index)
                                .context("Getting other participant for text inferrence.")
                                .unwrap();
                            let model_config_override = if let Some(cfg_ovr) = &other_char.1 {
                                Some(cfg_ovr.to_owned())
                            } else {
                                None
                            };

                            let context = TextInferenceContext {
                                character: other_char.0.clone(),
                                model_config_override,
                                chatlog_owner: self.character.clone(),
                                other_participants: self.other_participants.clone(),
                                chatlog: self.chatlog.clone(),
                                should_continue: false,
                                parameters: self.current_parameters.clone(),
                            };
                            self.show_progress_bar(context.character.clone());
                            let msg = llm_engine::LlmEngineRequest::TextInference(context);
                            if let Err(err) = self.send_to_server.send(msg) {
                                log::error!("Error during text infer additional request: {}", err);
                            }
                        } else {
                            log::debug!("No other participants defined for generation.");
                        }
                    }
                    _ => {}
                };
            }
        }

        ProcessInputResult::None
    }

    fn render_editing_parameters_modal(&self, frame: &mut Frame) {
        let mut area = centered_rect(60, 30, frame.size());
        area.height = std::cmp::min(area.height, 9);

        let mut hyperparameter_strings =
            vec![Line::from(format!("\"{}\"", self.current_parameters.name))
                .alignment(Alignment::Center)];

        if let Some(mirostat) = self.current_parameters.mirostat {
            if mirostat == 1 || mirostat == 2 {
                hyperparameter_strings.push(Line::from(format!("Mirostat {}", mirostat)));
                if let Some(eta) = self.current_parameters.mirostat_eta {
                    hyperparameter_strings.push(Line::from(format!("Eta: {}", eta)));
                }
                if let Some(tau) = self.current_parameters.mirostat_tau {
                    hyperparameter_strings.push(Line::from(format!("Tau: {}", tau)));
                }
            }
        } else {
            if let Some(top_k) = self.current_parameters.top_k {
                hyperparameter_strings.push(Line::from(format!("top k: {}", top_k)));
            }
            if let Some(top_p) = self.current_parameters.top_p {
                hyperparameter_strings.push(Line::from(format!("top p: {}", top_p)));
            }
            if let Some(min_p) = self.current_parameters.min_p {
                hyperparameter_strings.push(Line::from(format!("min p: {}", min_p)));
            }
            if let Some(temp) = self.current_parameters.temperature {
                hyperparameter_strings.push(Line::from(format!("temperature: {}", temp)));
            }
        };

        if let Some(repeat_penalty) = self.current_parameters.repeat_penalty {
            hyperparameter_strings.push(Line::from(format!("repeat penalty: {}", repeat_penalty)));
        }
        if let Some(repeat_range) = self.current_parameters.repeat_penalty_range {
            hyperparameter_strings.push(Line::from(format!("repeat range: {}", repeat_range)));
        }
        if let Some(frequency_penalty) = self.current_parameters.frequency_penalty {
            hyperparameter_strings.push(Line::from(format!(
                "frequency penalty: {}",
                frequency_penalty
            )));
        }

        let textarea = Paragraph::new(hyperparameter_strings)
            .style(Style::default().fg(Color::Cyan))
            .block(
                Block::default()
                    .title("Hyperparameters")
                    .borders(Borders::ALL),
            );

        frame.render_widget(Clear, area);
        frame.render_widget(textarea, area);
    }

    // render a single ChatLogItem to a VecDeque of ratatui Lines.
    fn render_chatlog_item(&self, area: Rect, chatlogitem: &ChatLogItem) -> VecDeque<Line<'_>> {
        // the bool keeps track of whether or not we're in a quote and
        // the chunker string is a buffer used so that we don't create
        // hundreds of strings in the loop.
        let mut in_quotes_state = false;
        let mut quote_chunker = String::new();

        // setup the styles depending on who's talking
        let mut text_style = Style::default();
        let mut quotes_style = Style::default();
        let mut name_style = Style::default();
        // check to see if this is from a character
        if chatlogitem
            .entity
            .eq_ignore_ascii_case(self.character.name.as_str())
        {
            if let Some(rgbs) = &self.character.name_rgb {
                name_style = name_style.fg(Color::Rgb(rgbs[0], rgbs[1], rgbs[2]));
            }
            if let Some(rgbs) = &self.character.text_rgb {
                text_style = text_style.fg(Color::Rgb(rgbs[0], rgbs[1], rgbs[2]));
            }
            if let Some(rgbs) = &self.character.quotes_rgb {
                quotes_style = quotes_style.fg(Color::Rgb(rgbs[0], rgbs[1], rgbs[2]));
            }
        }
        // or if this is from the user
        else if chatlogitem
            .entity
            .eq_ignore_ascii_case(&self.config.display_name.as_str())
        {
            if let Some(rgbs) = &self.config.display_name_rgb {
                name_style = name_style.fg(Color::Rgb(rgbs[0], rgbs[1], rgbs[2]));
            }
            if let Some(rgbs) = &self.config.text_rgb {
                text_style = text_style.fg(Color::Rgb(rgbs[0], rgbs[1], rgbs[2]));
            }
            if let Some(rgbs) = &self.config.quotes_rgb {
                quotes_style = quotes_style.fg(Color::Rgb(rgbs[0], rgbs[1], rgbs[2]));
            }
        }

        // check to see if other participants are loaded and if they have color syntax rules
        for other in &self.other_participants {
            if other
                .0
                .name
                .eq_ignore_ascii_case(chatlogitem.entity.as_str())
            {
                if let Some(rgbs) = &other.0.name_rgb {
                    name_style = name_style.fg(Color::Rgb(rgbs[0], rgbs[1], rgbs[2]));
                }
                if let Some(rgbs) = &other.0.text_rgb {
                    text_style = text_style.fg(Color::Rgb(rgbs[0], rgbs[1], rgbs[2]));
                }
                if let Some(rgbs) = &other.0.quotes_rgb {
                    quotes_style = quotes_style.fg(Color::Rgb(rgbs[0], rgbs[1], rgbs[2]));
                }
            }
        }

        // each log item may have multiple lines
        let item_lines = &chatlogitem.lines;
        let mut cli_lines_buffer: VecDeque<Line<'_>> = VecDeque::new();
        for (il_index, item_line) in item_lines.iter().enumerate() {
            // each line in the log item may be too long, so we break it apart
            let split_item_lines =
                slice_up_string(item_line, area.width as usize, chatlogitem.entity.len() + 2); // 2 == ": "
            for (si_index, split_item_line) in split_item_lines.iter().enumerate() {
                let mut spans = Vec::new();
                if il_index == 0 && si_index == 0 {
                    // for the first line of the chat log item we see if we have
                    // a known talker name, and color it differently
                    spans.push(Span::styled(
                        chatlogitem.entity.to_owned(),
                        name_style.bold(),
                    ));
                    spans.push(Span::styled(": ", text_style.bold()));
                }

                // Loop through the split line by graphemes and manually chunk things
                // up into quoted text and unquoted text.
                quote_chunker.clear();
                for g in UnicodeSegmentation::graphemes(split_item_line.as_str(), true) {
                    if g == "\"" {
                        if in_quotes_state {
                            quote_chunker.push_str(g);
                            spans.push(Span::styled(quote_chunker.to_owned(), quotes_style));
                            quote_chunker.clear();
                        } else {
                            spans.push(Span::styled(quote_chunker.to_owned(), text_style));
                            quote_chunker.clear();
                            quote_chunker.push_str(g);
                        }
                        in_quotes_state = !in_quotes_state;
                    } else {
                        quote_chunker.push_str(g);
                    }
                }
                // handle any left behind grapheme chunks
                if quote_chunker.is_empty() == false {
                    if in_quotes_state {
                        spans.push(Span::styled(quote_chunker.to_owned(), quotes_style));
                    } else {
                        spans.push(Span::styled(quote_chunker.to_owned(), text_style));
                    }
                }

                cli_lines_buffer.push_back(Line::from(spans));
            }
        }

        cli_lines_buffer
    }

    fn render_chatlog(&self, frame: &mut Frame, area: Rect) {
        // loop through the chat history and build up each line we want to render
        let mut chat_history: VecDeque<Line<'_>> = VecDeque::new();
        let lines_needed: usize = area.height as usize;

        for chatlogitem in self.chatlog.iter().rev().skip(self.chatlog_scroll) {
            let cli_lines_buffer = self.render_chatlog_item(area, chatlogitem);

            // add the chatlogitem lines into the chat history collection
            let mut maxed_out_lines = false;
            if self.config.display_chatlog_downward.unwrap_or_default() {
                for cli_line in cli_lines_buffer.into_iter().rev() {
                    if chat_history.len() >= lines_needed {
                        maxed_out_lines = true;
                        break;
                    }
                    chat_history.push_front(cli_line);
                }
            } else {
                for cli_line in cli_lines_buffer.into_iter() {
                    if chat_history.len() >= lines_needed {
                        maxed_out_lines = true;
                        break;
                    }
                    chat_history.push_back(cli_line);
                }
            }

            if maxed_out_lines {
                break;
            }

            // potentially add a buffer line if configured to do so
            if let Some(add_divider) = self.config.add_visual_buffer_between_chatlog_items {
                if add_divider && chat_history.len() < lines_needed {
                    if self.config.display_chatlog_downward.unwrap_or_default() {
                        chat_history.push_front(Line::from(" "));
                    } else {
                        chat_history.push_back(Line::from(" "));
                    }
                }
            }
        }

        // use the configured text alignment here.
        let alignment = if let Some(justification) = &self.config.chat_text_justification {
            justification.clone().into()
        } else {
            Alignment::Right
        };
        let chatlog = Paragraph::new(Vec::from(chat_history)).alignment(alignment);
        frame.render_widget(chatlog, area);
    }

    fn render_progress_bar(&mut self, frame: &mut Frame, area: Rect) {
        // lets create the widget if we haven't already
        if self.progress_widget.is_none() {
            let mut primary = self.config.progress_primary_rgb.unwrap_or([10, 242, 10]);
            let secondary = self.config.progress_secondary_rgb.unwrap_or([62, 62, 62]);

            // check to see if the character we're waiting on has an rgb value set for the name
            // and if so, use that for the primary color
            if let Some(char) = &self.waiting_for_character {
                if let Some(rgb) = char.name_rgb {
                    primary = rgb;
                }
            }

            let new_pw = ProgressBarScopeSignal::new(primary, secondary);
            self.progress_widget = Some(new_pw);
        }

        let pw = self.progress_widget.as_mut().unwrap();
        pw.render(frame, area);
    }

    // tells the UI to show the progress bar on next render
    fn show_progress_bar(&mut self, char_to_wait_on: CharacterFileYaml) {
        self.waiting_for_character = Some(char_to_wait_on);
        self.waiting_for_operation = true;
    }

    // tells the UI to no longer show the progress bar and free the widget
    fn hide_progress_bar(&mut self) {
        self.waiting_for_operation = false;
        self.progress_widget = None;
        self.waiting_for_character = None;
    }

    // a helper function to return the index into the chatlog for the currently
    // selected item. barely more space efficient than typing the code out...
    fn get_currently_select_chatlogitem_index(&self) -> usize {
        self.chatlog.len() - self.chatlog_scroll - 1
    }

    // main function to call to process any 'slash commands' the user types in
    fn process_slash_command(&mut self, user_cmd_str: &str) {
        // sanity checks
        if user_cmd_str.is_empty() {
            return;
        }

        // break the commands up by whitespace
        let cmd_split: Vec<&str> = user_cmd_str.split_whitespace().collect();
        if cmd_split.len() < 1 {
            return;
        }

        let mut character_selection: u8 = 1;
        let mut params: Vec<&str> = Vec::new();

        // if the next thing in the split is a number, use that as the character
        // selector to be active for the command
        if cmd_split.len() > 1 {
            if let Ok(n) = cmd_split[1].parse::<u8>() {
                character_selection = n.max(1);
                if cmd_split.len() > 2 {
                    params = cmd_split[2..].to_vec();
                }
            } else {
                params = cmd_split[1..].to_vec()
            }
        }

        // get the main command to be executed
        let main_cmd = cmd_split[0];

        match main_cmd {
            "get" => self.process_slash_command_get(character_selection, params),
            "set" => self.process_slash_command_set(character_selection, params),
            _ => {
                self.modal_messagebox = Some(MessageBoxModalWidget::new(
                    "Error",
                    format!("Unrecognized command: '{}'", main_cmd).as_str(),
                    60,
                    30,
                ));
            }
        }
    }

    // process the slash command: "get <char_num>? <var>"
    // return the result of the variable in a message box
    fn process_slash_command_get(&mut self, char_selector: u8, params: Vec<&str>) {
        // sanity checks
        if params.len() < 1 {
            self.modal_messagebox = Some(MessageBoxModalWidget::new(
                "Error",
                "Variable required. Syntax: 'get <char_num>? <variable>'",
                60,
                30,
            ));
            return;
        }

        // pull the character to be used for reference when accessing variables.
        // char_selector of 1 (or 0) is the active character, and the rest refer to other_participants
        // the odd ordering is meant to mimic the keybinding of triggering character responses
        let character = if char_selector < 2 {
            &self.character
        } else {
            let other_index = char_selector as usize - 2;
            if self.other_participants.len() > other_index {
                &self.other_participants[other_index].0
            } else {
                self.modal_messagebox = Some(MessageBoxModalWidget::new(
                    "Error",
                    format!("Cmd 'get'({}): unrecognized character index", char_selector).as_str(),
                    60,
                    30,
                ));
                return;
            }
        };

        let var_name = params[0];
        match var_name {
            "emotional_boosts" | "eb" => {
                let val = character
                    .emotional_boosts
                    .clone()
                    .unwrap_or("<no value>".to_string());

                // we logged the slash command request, so now log the returned value
                log::debug!("slash command 'get' returned: {}", val);

                self.modal_messagebox = Some(MessageBoxModalWidget::new(
                    "Information",
                    format!("'{}'({}): {}", var_name, char_selector, val).as_str(),
                    60,
                    30,
                ));
            }
            _ => {
                self.modal_messagebox = Some(MessageBoxModalWidget::new(
                    "Error",
                    format!(
                        "Cmd 'get'({}): unrecognized variable: {}",
                        char_selector, var_name
                    )
                    .as_str(),
                    60,
                    30,
                ));
            }
        }
    }

    // process the slash command: "set <char_num>? <var> ..."
    // sets the value for the variable specified to the rest of the input
    fn process_slash_command_set(&mut self, char_selector: u8, params: Vec<&str>) {
        // sanity checks
        if params.len() < 1 {
            self.modal_messagebox = Some(MessageBoxModalWidget::new(
                "Error",
                "Variable required. Syntax: 'set <char_num>? <variable> ...'",
                60,
                30,
            ));
            return;
        }

        // pull the character to be used for reference when accessing variables.
        // char_selector of 0 is the active character, and the rest refer to other_participants
        let character = if char_selector < 2 {
            &mut self.character
        } else {
            let other_index = char_selector as usize - 2;
            if self.other_participants.len() > other_index {
                &mut self.other_participants[other_index].0
            } else {
                self.modal_messagebox = Some(MessageBoxModalWidget::new(
                    "Error",
                    format!("Cmd 'set'({}): unrecognized character index", char_selector).as_str(),
                    60,
                    30,
                ));
                return;
            }
        };

        let var_name = params[0];
        match var_name {
            "emotional_boosts" | "eb" => {
                let val = if params.len() <= 1 {
                    None
                } else {
                    Some(params[1..].join(" "))
                };
                character.emotional_boosts = val;
            }
            _ => {
                self.modal_messagebox = Some(MessageBoxModalWidget::new(
                    "Error",
                    format!(
                        "Cmd 'set'({}): unrecognized variable: {}",
                        char_selector, var_name
                    )
                    .as_str(),
                    60,
                    30,
                ));
            }
        }
    }
}

impl TerminalRenderable for ChatState {
    fn process_input(&mut self, event: TerminalEvent) -> ProcessInputResult {
        // make sure to check for incoming message from the LLM engine
        self.process_incoming_llm_engine_messages();

        let mut result = ProcessInputResult::None;
        let index = self.get_currently_select_chatlogitem_index();

        if let Some(msgbox) = self.modal_messagebox.as_mut() {
            msgbox.process_input(event);
            if msgbox.is_finished {
                self.modal_messagebox = None;
            }
        } else if let Some((editor_type, editor)) = self.editor_widget.as_mut() {
            editor.process_input(event);
            if editor.is_finished {
                if editor.is_success {
                    match editor_type {
                        ChatEditorState::ChatlogItem => {
                            // if the editted string is empty, then just remove the chatlogitem
                            if editor.text.is_empty() {
                                self.chatlog.remove(index);
                            } else {
                                // if we made an edit, replace the strings in the chatlogitem
                                // and then attempt to save the logfile to secure the edits
                                if let Some(cli) = self.chatlog.get_mut(index) {
                                    cli.replace_items_with_string(editor.text.to_string());
                                } else {
                                    log::error!("Failed to update the log after editing a chatlog item. No change has been made in the log.");
                                }
                            }

                            if !self.save_chatlog_to_last_used() {
                                log::error!(
                                    "Failed to save the chatlog to the last used file ({:?}) after an edit",
                                    self.chatlog.get_last_used_filepath()
                                );
                            }
                        }

                        ChatEditorState::ChatlogContext => {
                            self.chatlog.current_context = editor.text.to_owned();

                            // attempt to save the changes to the chatlog
                            if !self.save_chatlog_to_last_used() {
                                log::error!("Failed to save the chatlog to the last used file ({:?}) after editing the context.", 
                                    self.chatlog.get_last_used_filepath());
                            }
                        }

                        ChatEditorState::UserDescription => {
                            if editor.text.is_empty() {
                                self.chatlog.user_description = None;
                            } else {
                                self.chatlog.user_description = Some(editor.text.to_owned());
                            }

                            // attempt to save the changes to the chatlog
                            if !self.save_chatlog_to_last_used() {
                                log::error!("Failed to save the chatlog to the last used file ({:?}) after editing the user description.", 
                                    self.chatlog.get_last_used_filepath());
                            }
                        }

                        ChatEditorState::SlashCommand => {
                            if !editor.text.is_empty() {
                                let user_command = editor.text.to_owned();
                                log::debug!(
                                    "User requested the following slash command: {}",
                                    user_command
                                );
                                self.process_slash_command(user_command.as_str());
                            }
                        }
                    }
                }
                self.editor_widget = None;
            }
        } else if self.editing_parameters {
            self.process_input_for_editing_parameters(event);
        } else if self.editing_reply {
            self.process_input_for_editing_replies(event)
        } else {
            result = self.process_input_for_viewing_chatlog(event);
        }

        result
    }

    fn render(&mut self, frame: &mut Frame) {
        frame.render_widget(Clear, frame.size());

        // use 80% of the frame up to the max width
        let hchunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(
                [
                    Constraint::Percentage(10),
                    Constraint::Percentage(80),
                    Constraint::Percentage(10),
                ]
                .as_ref(),
            )
            .split(frame.size());

        let chatlog_widget_width: usize = hchunks[1].width as usize;

        // build up the reply we're editing into a list of strings our column size
        let mut editing_reply_lines = vec![];
        if self.editing_reply {
            if !self.reply_text.is_empty() {
                // we don't add our name here, so leading space can be 0
                for reply_line in self.reply_text.lines() {
                    let split_lines = slice_up_string(reply_line, chatlog_widget_width, 0);
                    for split_line in split_lines {
                        editing_reply_lines.push(Line::from(split_line));
                    }
                }
            } else {
                editing_reply_lines.push(Line::from(vec![Span::styled(
                    "<Type Reply Here>",
                    Style::default().fg(Color::Rgb(100, 100, 100)),
                )]));
            }
            editing_reply_lines.push(Line::from("-".repeat(chatlog_widget_width)));
        }

        // start to budget how much space we need in that first row
        let editing_vertical_size = if self.waiting_for_operation {
            if let Some(widget) = &self.progress_widget {
                widget.get_requested_widget_height()
            } else {
                3 // assume there's some space needed
            }
        } else {
            editing_reply_lines.len() as u16
        };

        // do a rendering of the text that's currently in progress of being predicted
        let mut in_flight_lines: VecDeque<Line<'_>> = VecDeque::new();
        {
            if let Some(in_flight_text) = self.in_flight_text.as_ref() {
                // the temporary chatlogitem is created so that there's consistent rendering
                let in_flight_chatlogitem = ChatLogItem {
                    entity: self.waiting_for_character.as_ref().unwrap().name.clone(),
                    lines: in_flight_text
                        .trim_start()
                        .lines()
                        .map(|l| l.to_string())
                        .collect(),
                    ..Default::default()
                };
                in_flight_lines = self.render_chatlog_item(hchunks[1], &in_flight_chatlogitem);
            }
        }
        let in_flight_size = if in_flight_lines.is_empty()
            || self.config.disable_response_streaming.unwrap_or_default()
        {
            0
        } else {
            in_flight_lines.len() as u16 + 1
        };

        // do the layout for the main column. doing things like this allow for
        // different layouts depending on the chatlog flow direction.
        let (vchunks, progress_idx, in_flight_idx, chatlog_idx) =
            if self.config.display_chatlog_downward.unwrap_or_default() {
                let chatlog_buffer = if in_flight_lines.is_empty()
                    || self.progress_widget.is_some()
                    || self.editing_reply
                {
                    1
                } else {
                    0
                };

                (
                    Layout::default()
                        .direction(Direction::Vertical)
                        .constraints(
                            [
                                Constraint::Min(4),
                                Constraint::Max(chatlog_buffer),
                                Constraint::Max(in_flight_size),
                                Constraint::Max(editing_vertical_size),
                            ]
                            .as_ref(),
                        )
                        .split(hchunks[1]),
                    3,
                    2,
                    0,
                )
            } else {
                (
                    Layout::default()
                        .direction(Direction::Vertical)
                        .constraints(
                            [
                                Constraint::Max(editing_vertical_size),
                                Constraint::Max(in_flight_size),
                                Constraint::Min(4),
                            ]
                            .as_ref(),
                        )
                        .split(hchunks[1]),
                    0,
                    1,
                    2,
                )
            };

        // render the in flight text currently being predicted
        if !in_flight_lines.is_empty() {
            let alignment = if let Some(justification) = &self.config.chat_text_justification {
                justification.clone().into()
            } else {
                Alignment::Right
            };
            let in_flight_p = Paragraph::new(Vec::from(in_flight_lines)).alignment(alignment);
            frame.render_widget(in_flight_p, vchunks[in_flight_idx]);
        }

        // render either the reply editing or a progress bar
        if self.editing_reply {
            let alignment = if let Some(justification) = &self.config.chat_text_justification {
                justification.clone().into()
            } else {
                Alignment::Right
            };
            let editing_reply_p = Paragraph::new(editing_reply_lines).alignment(alignment);
            frame.render_widget(editing_reply_p, vchunks[progress_idx]);
        } else if self.waiting_for_operation {
            self.render_progress_bar(frame, vchunks[progress_idx]);
        }

        // render the visible portions of the chatlog
        self.render_chatlog(frame, vchunks[chatlog_idx]);

        // Now render any modal boxes over the chat log, only selecting one of them to draw.
        // This *should* mimic the same order that input processing gets called so that
        // there's no confusion.

        if let Some(msgbox) = &self.modal_messagebox {
            msgbox.render(frame);
        } else if let Some((_, editor)) = &self.editor_widget {
            editor.render(frame);
        }
        // if we're showing the parameters, create a new frame for it.
        else if self.editing_parameters {
            self.render_editing_parameters_modal(frame);
        }
    }
}

struct Lerper {
    first: f64,
    last: f64,
    start_time: Instant,
    duration_s: f64,
    bounce: bool,
    going_forward: bool,
}
impl Lerper {
    fn new(first: f64, last: f64, duration_s: f64, bounce: bool) -> Self {
        let now = Instant::now();
        Self {
            first,
            last,
            start_time: now,
            duration_s,
            bounce,
            going_forward: true,
        }
    }

    fn get(&mut self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let normalized = elapsed / self.duration_s;
        let ratio = normalized.min(1.0);

        let lerped = if self.going_forward {
            self.first + (ratio * (self.last - self.first))
        } else {
            self.last - (ratio * (self.last - self.first))
        };

        // check to see if we need to bounce the other way
        if self.bounce && elapsed >= self.duration_s {
            self.going_forward = !self.going_forward;
            self.start_time += Duration::from_secs_f64(self.duration_s);
        }

        lerped
    }
}

// A simple progress bar widget based on randomized sparkline data
struct ProgressBarScopeSignal {
    data_buffer1: Vec<(f64, f64)>,
    data_buffer2: Vec<(f64, f64)>,
    start_time: Instant,
    speed: f64,
    freq_lerp1: Lerper,
    freq_lerp2: Lerper,
    primary_rgb: [u8; 3],
    secondary_rgb: [u8; 3],
}
impl ProgressBarScopeSignal {
    fn new(primary_rgb: [u8; 3], secondary_rgb: [u8; 3]) -> Self {
        const SPEED_VARIABILITY: f64 = 1.0;

        let start_time = Instant::now();
        let mut rng = rand::thread_rng();
        let speed = 8.0 + rng.gen_range(-SPEED_VARIABILITY..SPEED_VARIABILITY);

        Self {
            data_buffer1: Vec::with_capacity(256),
            data_buffer2: Vec::with_capacity(256),
            start_time,
            speed,
            freq_lerp1: Lerper::new(2.0, 10.0, 4.0, true),
            freq_lerp2: Lerper::new(1.314, 2.17, 10.31, true),
            primary_rgb,
            secondary_rgb,
        }
    }

    // should return the number of rows requested for layout of this widget
    fn get_requested_widget_height(&self) -> u16 {
        5
    }

    fn generate_2d_sin_waves(
        buffer: &mut Vec<(f64, f64)>,
        amplitude: f64,
        frequency: f64,
        phase: f64,
        num_points: usize,
    ) {
        buffer.clear();

        for i in 0..num_points {
            let x = i as f64 / num_points as f64; // normalized time
            let y = amplitude * (2.0 * std::f64::consts::PI * frequency * x + phase).sin();
            buffer.push((x, y));
        }
    }

    fn render(&mut self, frame: &mut Frame, area: Rect) {
        // update the data buffer
        let t = self.speed * self.start_time.elapsed().as_secs_f64();

        let freq1 = self.freq_lerp1.get();
        Self::generate_2d_sin_waves(&mut self.data_buffer1, 1.0, freq1, t, area.width as usize);
        let freq2 = self.freq_lerp2.get();
        Self::generate_2d_sin_waves(&mut self.data_buffer2, 0.8, freq2, t, area.width as usize);

        let dataset = vec![
            Dataset::default()
                .marker(ratatui::symbols::Marker::Dot)
                .style(Style::default().fg(Color::Rgb(
                    self.secondary_rgb[0],
                    self.secondary_rgb[1],
                    self.secondary_rgb[2],
                )))
                .graph_type(ratatui::widgets::GraphType::Scatter)
                .data(&self.data_buffer1),
            Dataset::default()
                .marker(ratatui::symbols::Marker::Dot)
                .style(Style::default().fg(Color::Rgb(
                    self.primary_rgb[0],
                    self.primary_rgb[1],
                    self.primary_rgb[2],
                )))
                .graph_type(ratatui::widgets::GraphType::Scatter)
                .data(&self.data_buffer2),
        ];

        let scope = Chart::new(dataset)
            .x_axis(ratatui::widgets::Axis::default().bounds([0.0, 1.0]))
            .y_axis(ratatui::widgets::Axis::default().bounds([-1.0, 1.0]));

        frame.render_widget(scope, area);
    }
}

// A simple progress bar widget based on randomized sparkline data
#[allow(dead_code)]
struct ProgressBarRandomSparkline {
    tick_rate: Duration,
    last_tick: Instant,
    signal: Vec<u64>,
    area: Rect,
    rng: ThreadRng,
}
impl ProgressBarRandomSparkline {
    #[allow(dead_code)]
    fn new(tick_rate: Duration, area: Rect) -> Self {
        let mut rng = rand::thread_rng();
        let signal: Vec<u64> = (0..area.width).map(|_| rng.gen_range(0..100)).collect();

        Self {
            tick_rate,
            last_tick: Instant::now(),
            signal,
            area,
            rng,
        }
    }

    // checks tho see if the progress bar should be updated
    #[allow(dead_code)]
    fn tick(&mut self) {
        if self.tick_rate < self.last_tick.elapsed() {
            self.signal.pop();
            let next = self.rng.gen_range(0..100);
            self.signal.insert(0, next);
            self.last_tick += self.tick_rate;
        }
    }

    // render the progress bar in the user interface, and will
    // automatically adjust the internal structures to fit the
    // area passed in.
    #[allow(dead_code)]
    fn render(&mut self, frame: &mut Frame, area: Rect) {
        // check to see if the UI has been resized since creation
        if area.width != self.area.width {
            if area.width > self.area.width {
                let delta = area.width - self.area.width;
                for _ in 0..delta {
                    let next = self.rng.gen_range(0..100);
                    self.signal.insert(0, next);
                }
            } else {
                let delta = self.area.width - area.width;
                for _ in 0..delta {
                    self.signal.pop();
                }
            }
            self.area = area;
        }

        let sparkline = Sparkline::default().data(&self.signal);
        frame.render_widget(sparkline, area);
    }
}
