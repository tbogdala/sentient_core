use std::time::{Duration, Instant};

use anyhow::Result;

use crate::{
    character_select::CharacterSelectState,
    chat::ChatState,
    chatlog::ChatLog,
    config::{CharacterFileYaml, ConfigurationFile},
    llm_engine,
    log_select::LogSelectState,
    main_menu::MainMenuState,
    tui::{ProcessInputResult, Tui},
};

// This enumeration indicates what scene is active in the application.
#[derive(PartialEq)]
pub enum ApplicationState {
    MainMenu,
    CharacterSelect,
    CharacterLogSelect(CharacterFileYaml),
    Chat(CharacterFileYaml, ChatLog),
}

// This is the main application state object for the app.
pub struct Application<'a> {
    // this is the terminal abstraction used to hide implementation details
    // away from the application.
    terminal: &'a mut Tui,

    // our active configuration file for the app, loaded
    config: ConfigurationFile,

    // the LLM engine worker thread controller
    pub engine: llm_engine::LlmEngine,

    // an enum indicating which state is active in the application
    current_state: ApplicationState,

    // contains the main menu scene's state
    mainmenu_state: MainMenuState,

    // contains the character select scene's state
    character_select_state: Option<CharacterSelectState>,

    // optionally contains the log selector scene's state
    log_select_state: Option<LogSelectState>,

    // optionally contains the chat scene's state
    chat_state: Option<ChatState>,
}
impl<'a> Application<'a> {
    // Creates a new Application object.
    pub fn new(
        terminal: &'a mut Tui,
        config: ConfigurationFile,
        engine: llm_engine::LlmEngine,
    ) -> Application<'a> {
        Application {
            terminal,
            config,
            engine,
            current_state: ApplicationState::MainMenu,
            mainmenu_state: MainMenuState::default(),
            character_select_state: None,
            log_select_state: None,
            chat_state: None,
        }
    }

    // Runs the application loop that draws the current application state and then
    // processes the input.
    pub fn run(&mut self, ui_draw_tick_rate: u64) -> Result<()> {
        let draw_tick_rate = Duration::from_millis(ui_draw_tick_rate);
        let mut draw_last_tick = Instant::now();
        loop {
            let perform_draw: bool = draw_tick_rate < draw_last_tick.elapsed();
            let mut proc_result = ProcessInputResult::None;

            match self.current_state {
                ApplicationState::MainMenu => {
                    if perform_draw {
                        self.terminal
                            .draw(&mut self.mainmenu_state)
                            .expect("failed to draw the main menu UI");
                    }
                    proc_result = self.terminal.process_input(&mut self.mainmenu_state);
                }
                ApplicationState::CharacterSelect => {
                    if let Some(charselect) = self.character_select_state.as_mut() {
                        if perform_draw {
                            self.terminal
                                .draw(charselect)
                                .expect("failed to draw the character select UI");
                        }
                        proc_result = self.terminal.process_input(charselect);
                    }
                }
                ApplicationState::CharacterLogSelect(_) => {
                    if let Some(logselect) = self.log_select_state.as_mut() {
                        if perform_draw {
                            self.terminal
                                .draw(logselect)
                                .expect("failed to draw the log selector UI");
                        }
                        proc_result = self.terminal.process_input(logselect);
                    }
                }
                ApplicationState::Chat(_, _) => {
                    if let Some(chat_state) = self.chat_state.as_mut() {
                        if perform_draw {
                            self.terminal
                                .draw(chat_state)
                                .expect("failed to draw the chat UI");
                        }
                        proc_result = self.terminal.process_input(chat_state);
                    }
                }
            };

            if perform_draw {
                draw_last_tick += draw_tick_rate;
            }

            // Based on what the current scene decides, possibly take an action
            match proc_result {
                ProcessInputResult::Quit => {
                    return Ok(());
                }
                ProcessInputResult::ChangeScene(new_scene) => {
                    // mark the new scene as current
                    self.current_state = new_scene;

                    // create the new state object if needed
                    match &self.current_state {
                        ApplicationState::MainMenu => {}
                        ApplicationState::CharacterSelect => {
                            self.character_select_state = Some(CharacterSelectState::new());
                        }
                        ApplicationState::CharacterLogSelect(chararcter) => {
                            self.log_select_state =
                                Some(LogSelectState::new(chararcter.clone(), self.config.clone()));
                        }
                        ApplicationState::Chat(character, chatlog) => {
                            let params = self.config.parameters.first();
                            self.chat_state = Some(ChatState::new(
                                character.to_owned(),
                                chatlog.to_owned(),
                                params,
                                self.config.clone(),
                                self.engine.send_to_server.clone(),
                                self.engine.send_cmd_to_server.clone(),
                                self.engine.recv_on_client.clone(),
                            ));
                        }
                    }
                }
                ProcessInputResult::None => {}
            }

            // put the loop to sleep
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}
