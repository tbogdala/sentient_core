use anyhow::{Context, Result};
use application::Application;

use llm_engine::{LlmEngine, LlmEngineResponse};
use simple_logger::SimpleLogger;
use tui::Tui;

mod application;
mod character_select;
mod chat;
mod chatlog;
mod config;
mod llm_engine;
mod log_select;
mod main_menu;
mod tui;

#[cfg(feature = "sentence_similarity")]
mod vector_embedding_engine;

// This is how long the timeout should be in milliseconds for the terminal's backend
const INPUT_THREAD_READ_TIMEOUT_MS: u64 = 1000 / 4;
const UI_DRAW_TICK_RATE: u64 = 1000 / 30;

fn main() -> Result<()> {
    // parse the command-line arguments
    let cmd_arg_matches = clap::Command::new("sentient_core")
        .about("sentient_core: a terminal interface to AI characters.")
        .arg(clap::Arg::new("config-file")
            .short('c')
            .long("config-file")
            .default_value("config.yaml")
            .action(clap::ArgAction::Set)
                .value_name("FILE")
                .help("Specifies the configuration file to load instead of config.yaml."))
        .arg(
            clap::Arg::new("model-file-or-name")
                .short('m')
                .long("model-file-or-name")
                .action(clap::ArgAction::Set)
                .value_name("FILE")
                .help("The model to load to chat with. Either configured name or filepath of the model are acceptable."),
        )
        .arg_required_else_help(true)
        .get_matches();

    SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .env()
        .with_colors(true)
        .init()
        .unwrap();

    // ***********************************************************************
    // load the configuration file for the application.
    let custom_config_filename: Option<&String> = cmd_arg_matches.get_one::<String>("config-file");
    if !std::path::Path::new(custom_config_filename.unwrap()).exists() {
        println!("The 'config.yaml' configuration file is missing. Place it next to the application or specify its location with the -c argument.");
        std::process::exit(1);
    }

    let config = config::ConfigurationFile::load_config(custom_config_filename);

    // ***********************************************************************
    // Spawn the LLM Engine thread.
    // take care of the LLM loading right away, panic if things fail right now.
    let model_fileorname_p = cmd_arg_matches.get_one::<String>("model-file-or-name");
    let model_fileorname = if model_fileorname_p.is_none() {
        if config.models.is_empty() {
            println!("A model must be configured in 'config.yaml' before running the application.");
            std::process::exit(1);
        }
        println!("Consider specifying a model name on the command line with the -m argument.");
        println!(
            "Since no model was specified, the application will load the first configured model."
        );
        config.models[0].name.as_str()
    } else {
        model_fileorname_p.unwrap()
    };
    let engine = LlmEngine::spawn(config.clone(), model_fileorname.to_string());

    // wait here for the engine to respond.
    let res = engine
        .recv_on_client
        .recv()
        .expect("Main thread didn't like recv attempt for llm engine channels.");
    if res != LlmEngineResponse::ModelLoaded {
        log::error!(
            "First LlmEngineResponse wasn't model loaded. Suspect problems if that wasn't planned"
        )
    }

    // ***********************************************************************
    // setup the terminal and run the loop, hoping to restore terminal on exit.
    let mut tui = Tui::new(INPUT_THREAD_READ_TIMEOUT_MS)
        .context("failed to create the terminal interface")?;
    Tui::enable().context("should have been able to start the terminal interface")?;

    // **********************************************************************
    // run the actual app
    let mut app = Application::new(&mut tui, config.clone(), engine);
    if let Err(err) = app.run(UI_DRAW_TICK_RATE) {
        log::error!("Application loop failed: {err}")
    }

    // *******************************************************************
    // tell the server to shut down ... and try to wait for it to happen.
    let shutdown_req_result = app
        .engine
        .send_to_server
        .try_send(llm_engine::LlmEngineRequest::ImmediateShutdown);
    if shutdown_req_result.is_ok() {
        // the request went through so wait for the thread to close.
        // this avoids segfaulting on exit.
        let _ = app.engine.handle.join();
    } else if let Err(err) = shutdown_req_result {
        log::error!("Failed to shutdown the LLM server thread: {err}");
    }

    // ***************************************************************
    // restore the terminal now that the application is quitting.
    Tui::disable().context("failed to disable the terminal interface")?;

    Ok(())
}
