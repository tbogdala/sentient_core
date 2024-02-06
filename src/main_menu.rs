use crossterm::event::KeyCode;
use ratatui::{
    prelude::{Alignment, Constraint, Direction, Layout},
    style::Stylize,
    text::Line,
    widgets::Paragraph,
    Frame,
};

use crate::tui::{ProcessInputResult, TerminalEvent, TerminalRenderable};

#[derive(Default)]
pub struct MainMenuState {}
impl TerminalRenderable for MainMenuState {
    fn process_input(&mut self, event: TerminalEvent) -> ProcessInputResult {
        match event {
            TerminalEvent::Key(key) => {
                if key.code == KeyCode::Char('q') {
                    return ProcessInputResult::Quit;
                }
                if key.code == KeyCode::Char('c') {
                    return ProcessInputResult::ChangeScene(
                        crate::application::ApplicationState::CharacterSelect,
                    );
                }
            }
            _ => {}
        }

        ProcessInputResult::None
    }

    fn render(&mut self, frame: &mut Frame) {
        let main_title_seq = vec![
            Line::from("Sentient Core".bold()),
            Line::from("-------------"),
            Line::from("(c)hat"),
            Line::from(""),
            Line::from("(q)uit"),
        ];

        let hchunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(
                [
                    Constraint::Percentage(20),
                    Constraint::Percentage(60),
                    Constraint::Percentage(20),
                ]
                .as_ref(),
            )
            .split(frame.size());

        let vchunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(20), Constraint::Min(4)].as_ref())
            .split(hchunks[1]);

        let title = Paragraph::new(main_title_seq).alignment(Alignment::Center);
        frame.render_widget(title, vchunks[1]);
    }
}
