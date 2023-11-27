use anyhow::{Context, Result};
use crossbeam::channel::Receiver;
use crossterm::{
    event::{
        self, Event as CrosstermEvent, KeyCode, KeyEvent as CrosstermKeyEvent,
        MouseEvent as CrosstermMouseEvent,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    prelude::{Constraint, CrosstermBackend, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, ListState, Paragraph},
    Terminal,
};
use std::{
    io, panic, thread,
    time::{Duration, Instant},
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use crate::application::ApplicationState;

// Used to control application flow from the specialized input handlers
// for each ApplicationState scene.
#[derive(PartialEq)]
pub enum ProcessInputResult {
    // no action is needed
    None,

    // user requested the app to quit
    Quit,

    // user has requested a scene change
    ChangeScene(ApplicationState),
}

/// Both the event pump and the thin wrapper are reskins of the code fround in the Ratatui Book:
// ihttps://ratatui.rs/index.html
// This file isn't meant to be super independent, though I'm trying to keep it generic.

// Define some type aliases here to make things easier everywhere else. This application doesn't need
// to preserve stdin/stdout and can freely take it over, so we'll use that as our backend.
pub type Frame<'a> = ratatui::Frame<'a, ratatui::backend::CrosstermBackend<std::io::Stdout>>;
pub type CrosstermTerminal = ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>;

// Implement this on all UI state objects that can be drawn.
pub trait TerminalRenderable {
    fn render(&mut self, f: &mut Frame);
    fn process_input(&mut self, event: TerminalEvent) -> ProcessInputResult;
}

// A type encapsulating all the terminal events we wish to capture and report.
#[derive(Clone, Copy, Debug)]
pub enum TerminalEvent {
    // terminal tick
    Tick,
    // key press
    Key(CrosstermKeyEvent),
    // mouse click or scroll,
    Mouse(CrosstermMouseEvent),
    // terminal resize
    Resize(u16, u16),
}

pub struct TerminalEventHandler {
    // event receiver channel
    receiver: Receiver<TerminalEvent>,

    // event handler thread handle
    _handler: thread::JoinHandle<()>,
}
impl TerminalEventHandler {
    // Creates a new TerminalEventHandler with the specified tick rate in milliseconds.
    pub fn new(tick_rate: u64) -> Self {
        let tick_rate = Duration::from_millis(tick_rate);
        let (sender, receiver) = crossbeam::channel::unbounded();
        let _handler = {
            let sender = sender.clone();
            thread::spawn(move || {
                let mut last_tick = Instant::now();
                loop {
                    // use the tick_rate minus the elapsed time since last tick
                    // defaults to just tick_rate on overflow.
                    let timeout = tick_rate
                        .checked_sub(last_tick.elapsed())
                        .unwrap_or(tick_rate);
                    if event::poll(timeout).expect("should be able to poll terminal events") {
                        // We have an event to handle, so lets see if we're interested
                        let e = event::read()
                            .expect("should be able to read an event that poll() says exists");
                        match e {
                            CrosstermEvent::Key(e) =>
                            // we only pass on 'press' events for multiplatform compatibility
                            {
                                if e.kind == event::KeyEventKind::Press {
                                    sender.send(TerminalEvent::Key(e))
                                } else {
                                    Ok(())
                                }
                            }

                            CrosstermEvent::Mouse(e) => sender.send(TerminalEvent::Mouse(e)),
                            CrosstermEvent::Resize(w, h) => {
                                sender.send(TerminalEvent::Resize(w, h))
                            }

                            // ignore the rest
                            CrosstermEvent::FocusGained => Ok(()),
                            CrosstermEvent::FocusLost => Ok(()),
                            CrosstermEvent::Paste(_) => Ok(()),
                        }
                        .expect("failed to pass on the detected terminal event")
                    }

                    if last_tick.elapsed() >= tick_rate {
                        sender
                            .send(TerminalEvent::Tick)
                            .expect("failed to send the tick event");
                        last_tick = Instant::now();
                    }
                }
            }) //thread::spwn()
        };

        Self { receiver, _handler }
    }

    // attempts to get the next input and should return None if none exist.
    // as a backup, a timeout is created and the duration can be passed in milliseconds.
    pub fn get_next_input(&self, timeout_ms: Option<u64>) -> Option<TerminalEvent> {
        let timeout = Duration::from_millis(timeout_ms.unwrap_or(16));
        if self.receiver.is_empty() {
            None
        } else {
            match self.receiver.recv_timeout(timeout) {
                Ok(event) => Some(event),
                Err(err) => {
                    log::error!("Failed to receive the event on the input handler pump: {err}");
                    None
                }
            }
        }
    }
}

// This is a thin abstraction around the terminal interface.
// Note: the enable()/disable() functions don't need a self reference
// so they're kept as type functions so as they can be used more flexibly
// (e.g. panic hooks)
pub struct Tui {
    // the internal interface to the terminal
    terminal: CrosstermTerminal,

    // encapsulates the event management for the terminal
    pub events: TerminalEventHandler,

    // how frequently the input should be polled
    input_tick_rate_ms: u64,
}
impl Tui {
    // creates a new terminal interface that encapsulates the terminal ui backend
    // for the application.
    pub fn new(input_tick_rate_ms: u64) -> Result<Self> {
        let terminal = Terminal::new(CrosstermBackend::new(io::stdout()))
            .context("creating terminal backend interface failed")?;
        let events = TerminalEventHandler::new(input_tick_rate_ms);

        let panic_hook = panic::take_hook();
        panic::set_hook(Box::new(move |panic| {
            Self::disable().expect("failed to reset the terminal on detected panic");
            panic_hook(panic);
        }));

        Ok(Self {
            terminal,
            events,
            input_tick_rate_ms,
        })
    }

    // enables the terminal interface
    pub fn enable() -> Result<()> {
        enable_raw_mode().context("Failed to enable raw mode")?;
        execute!(io::stdout(), crossterm::terminal::EnterAlternateScreen)
            .context("unable to enter alternate screen")?;

        Ok(())
    }

    // disables the terminal interface
    pub fn disable() -> Result<()> {
        disable_raw_mode().context("failed to disable raw mode")?;
        execute!(io::stdout(), crossterm::terminal::LeaveAlternateScreen)
            .context("unable to switch to main screen")?;

        Ok(())
    }

    // draws the given frame to the terminal backend
    pub fn draw<T: TerminalRenderable>(&mut self, b: &mut T) -> Result<()> {
        self.terminal.draw(|frame| b.render(frame))?;
        Ok(())
    }

    pub fn process_input<T: TerminalRenderable>(&mut self, b: &mut T) -> ProcessInputResult {
        // read input until the processing function returns something that's not
        // ProcessInputResult::None or we're out of input.
        while let Some(terminal_event) = self.events.get_next_input(Some(self.input_tick_rate_ms)) {
            let result = b.process_input(terminal_event);
            if result != ProcessInputResult::None {
                return result;
            }
        }

        ProcessInputResult::None
    }
}

pub fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Percentage((100 - percent_y) / 2),
                Constraint::Percentage(percent_y),
                Constraint::Percentage((100 - percent_y) / 2),
            ]
            .as_ref(),
        )
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints(
            [
                Constraint::Percentage((100 - percent_x) / 2),
                Constraint::Percentage(percent_x),
                Constraint::Percentage((100 - percent_x) / 2),
            ]
            .as_ref(),
        )
        .split(popup_layout[1])[1]
}

// This is a generic list state, pulled from ratatui's list example, that
// can be used to build out our list selectors.
pub struct StatefulList<T> {
    pub state: ListState,
    pub items: Vec<T>,
}
impl<T> StatefulList<T> {
    pub fn with_items(items: Vec<T>) -> StatefulList<T> {
        StatefulList {
            state: ListState::default(),
            items,
        }
    }

    pub fn next(&mut self) {
        if !self.items.is_empty() {
            let i = match self.state.selected() {
                Some(i) => {
                    if i >= self.items.len() - 1 {
                        0
                    } else {
                        i + 1
                    }
                }
                None => 0,
            };
            self.state.select(Some(i));
        }
    }

    pub fn previous(&mut self) {
        if !self.items.is_empty() {
            let i = match self.state.selected() {
                Some(i) => {
                    if i == 0 {
                        self.items.len() - 1
                    } else {
                        i - 1
                    }
                }
                None => 0,
            };
            self.state.select(Some(i));
        }
    }

    // pub fn unselect(&mut self) {
    //     self.state.select(None);
    // }
}

pub struct TextEditingBlockModalWidget {
    // the title of the block when displaying the widget
    pub title: String,

    // the string to edit
    pub text: String,

    // should be set to true after `process_input()` when the user is done editing
    pub is_finished: bool,

    // should be set to true if the user 'accepted' the edits (false if they cancelled)
    // after `process_input()`.
    pub is_success: bool,
}
impl TextEditingBlockModalWidget {
    pub fn new(title: String, string_to_edit: String) -> Self {
        Self {
            title,
            text: string_to_edit,
            is_finished: false,
            is_success: false,
        }
    }

    pub fn process_input(&mut self, event: TerminalEvent) {
        if let TerminalEvent::Key(key) = event {
            match key.code {
                KeyCode::Esc => {
                    self.is_success = false;
                    self.is_finished = true;
                }
                KeyCode::Backspace => {
                    self.text.pop();
                }
                KeyCode::Char(to_insert) => {
                    self.text.push(to_insert);
                }
                KeyCode::Enter => {
                    self.is_success = true;
                    self.is_finished = true;
                }
                _ => {}
            }
        }
    }

    pub fn render(&self, frame: &mut Frame) {
        let mut area = centered_rect(80, 60, frame.size());

        // get the width to split the text by so that there's nice word wrapping
        let split_width = (area.width - 2) as usize;

        let mut editing_lines = vec![];
        if !self.text.is_empty() {
            let split_lines = slice_up_string(&self.text, split_width, 0);
            for split_line in split_lines {
                editing_lines.push(Line::from(split_line));
            }
        } else {
            editing_lines.push(Line::from(vec![Span::styled(
                "<Type Text Here>",
                Style::default().fg(Color::Rgb(100, 100, 100)),
            )]));
        }

        // make size the box to the number of lines + 1, accounting for the border
        area.height = std::cmp::min(area.height, 3 + editing_lines.len() as u16);

        let textarea = Paragraph::new(editing_lines).style(Style::default()).block(
            Block::default()
                .border_style(Style::default().fg(Color::Cyan))
                .title(self.title.as_str())
                .borders(Borders::ALL),
        );

        frame.render_widget(Clear, area);
        frame.render_widget(textarea, area);
    }
}

// A basic modal dialog box with a configurable title and body text.
pub struct MessageBoxModalWidget {
    // the title of the border on the modal box
    pub title: String,

    // the string to edit
    pub text: String,

    // should be set to true after `process_input()` when the user is done editing
    pub is_finished: bool,

    // the percentage of screen width to take up at max
    pub width_pct: u16,

    // the percentage of screen height to take up at max
    pub height_pct: u16,
}
impl MessageBoxModalWidget {
    pub fn new(title: &str, text: &str, width_pct: u16, height_pct: u16) -> Self {
        Self {
            title: title.to_string(),
            text: text.to_string(),
            is_finished: false,
            width_pct,
            height_pct,
        }
    }

    pub fn process_input(&mut self, event: TerminalEvent) {
        if let TerminalEvent::Key(key) = event {
            match key.code {
                KeyCode::Esc => {
                    self.is_finished = true;
                }
                KeyCode::Enter => {
                    self.is_finished = true;
                }
                _ => {}
            }
        }
    }

    pub fn render(&self, frame: &mut Frame) {
        let mut area = centered_rect(self.width_pct, self.height_pct, frame.size());

        // get the width to split the text by so that there's nice word wrapping
        let split_width = (area.width - 2) as usize;

        let mut msgbox_lines = vec![];
        if !self.text.is_empty() {
            let sentences = UnicodeSegmentation::split_sentence_bounds(self.text.as_str())
                .collect::<Vec<&str>>();
            for sentence in sentences {
                let split_lines = slice_up_string(sentence, split_width, 0);
                for split_line in split_lines {
                    msgbox_lines.push(Line::from(split_line));
                }
            }
        }

        // make size the box to the number of lines + 1, accounting for the border
        area.height = std::cmp::min(area.height, 2 + msgbox_lines.len() as u16);

        let textarea = Paragraph::new(msgbox_lines).style(Style::default()).block(
            Block::default()
                .border_style(Style::default().fg(Color::Cyan))
                .title(self.title.as_str())
                .borders(Borders::ALL),
        );

        frame.render_widget(Clear, area);
        frame.render_widget(textarea, area);
    }
}

// takes a reference to a String and generates a vector of new Strings
// that are at most 'max_width' long and are broken apart by whitespace.
// 'leading_space_reserve' makes the first line a little shorter, so that
// when the name is added later, it won't overflow the line.
//
// note: the function is intended to work intuitively with unicode data
// as well, so things use grapheme width for calculations and word breaks, etc...
pub fn slice_up_string(
    source: &str,
    max_width: usize,
    leading_space_reserve: usize,
) -> Vec<String> {
    // we start the current max_limit off lower, potentially, and then
    // return it to max_width after the first string has been split.
    let mut current_max_limit = max_width - leading_space_reserve;

    // return the slice if not necessary to split
    let source_width = UnicodeWidthStr::width(source);
    if source_width < current_max_limit {
        return vec![source.to_owned()];
    }

    let mut current_display_width = 0;
    let mut line_buffer: String = String::new();
    let mut accumulator = String::new();
    let mut result: Vec<String> = Vec::new();

    let mut grapheme_iter = source.split_word_bounds().peekable();
    let mut unicode_word = grapheme_iter.next();

    // we start off iterating by grapheme word
    while let Some(word) = unicode_word {
        let word_width = UnicodeWidthStr::width(word);
        // if that word is whitespace, dump the accumulator into the line buffer and
        // this word and clear the accumulator to start again.
        if word.chars().all(|c| c.is_whitespace()) {
            if accumulator.is_empty() == false {
                if word_width + current_display_width < current_max_limit {
                    line_buffer.push_str(accumulator.as_str());
                    line_buffer.push_str(word);
                    accumulator.clear();
                    current_display_width += word_width;
                } else {
                    // make a copy of the resulting string
                    result.push(line_buffer.trim_end().to_owned());
                    current_max_limit = max_width;

                    line_buffer.clear();
                    line_buffer.push_str(accumulator.as_str());
                    line_buffer.push_str(word);
                    accumulator.clear();
                    current_display_width = UnicodeWidthStr::width(line_buffer.as_str());
                }
            }
        } else {
            // grapheme clusters too big for the width might end up here, so handle them specially.
            if accumulator.is_empty() == true && word_width > current_max_limit {
                // push the line_buffer as a result first (note accumulator is tested to be empty)
                if line_buffer.is_empty() == false {
                    result.push(line_buffer.trim_end().to_owned());
                    line_buffer.clear();
                    current_max_limit = max_width;
                }

                let mut big_word_buffer = String::new();
                let mut big_word_grapheme_count = 0;
                for grapheme in word.graphemes(true) {
                    if big_word_grapheme_count < current_max_limit {
                        big_word_buffer.push_str(grapheme);
                        big_word_grapheme_count += 1;
                    }
                    if big_word_grapheme_count >= current_max_limit {
                        result.push(big_word_buffer.to_owned());
                        current_max_limit = max_width;
                        big_word_buffer.clear();
                        big_word_grapheme_count = 0;
                    }
                }
                if big_word_buffer.is_empty() == false {
                    accumulator.push_str(big_word_buffer.as_str());
                    current_display_width = UnicodeWidthStr::width(big_word_buffer.as_str());
                }
            } else {
                // normal grapheme word block logic ... accumulate until we hit whitespace
                accumulator.push_str(word);
                current_display_width += word_width;
            }
        }
        unicode_word = grapheme_iter.next();
    }
    // push the remaining fragment as a line if it fits into the line buffer
    // we don't have to check the width of the accumulator, because it's already been added to
    // the current_display_width
    if accumulator.is_empty() == false && current_display_width < current_max_limit {
        line_buffer.push_str(accumulator.as_str());
        accumulator.clear();
    }
    // push the remaining line out
    if line_buffer.is_empty() == false {
        result.push(line_buffer.trim_end().to_owned());
    }
    // truly the last fragment, the accumulator must not have fit within the current_max_limit,
    // so just add it as a new line.
    if accumulator.is_empty() == false {
        result.push(accumulator.to_string());
    }

    result
}
