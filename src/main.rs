use std::error::Error;
use std::io::{stdout, stdin, Write};
use std::time::Duration;
use std::path::{Path, PathBuf};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    prelude::*,
    Terminal,
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
};

// Represents the application modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppMode {
    SampleSelection,
    MainMenu,
    DirectoryPrompt, // New mode for directory prompt popup
}

// For popup states
#[derive(Debug)]
enum PopupState {
    Hidden,
    DirectoryInput {
        input: String,
        cursor_position: usize,
        message: String,
    },
    DirectoryConfirmation {
        directory: PathBuf,
        message: String,
    },
    DiagnosticsLog {
        log_entries: Vec<String>,
    },
}

// Define our app state
struct App {
    // Application mode
    mode: AppMode,
    // Sample selection state
    samples: Vec<PathBuf>,
    sample_selection_state: ListState,
    selected_sample: Option<PathBuf>,
    // Navigation state
    menu_stack: Vec<Menu>,
    current_menu_state: ListState,
    // Whether the app should exit
    should_quit: bool,
    // Popup state for UI input prompts
    popup: PopupState,
    // Directory where experiments are located
    experiment_dir: Option<PathBuf>,
    // Whether to show diagnostic logs
    show_diagnostics: bool,
}

// Define menu items
#[derive(Clone)]
struct MenuItem {
    title: String,
    // Either a submenu or an action
    kind: MenuItemKind,
}

#[derive(Clone)]
enum MenuItemKind {
    Submenu(Menu),
    Action(fn(&mut App)),
}

// Define a menu
#[derive(Clone)]
struct Menu {
    title: String,
    items: Vec<MenuItem>,
}

impl App {
    fn new(experiment_dir_option: Option<PathBuf>, show_diagnostics: bool) -> Self {
        // Create our menu structure

        // System Information submenu
        let system_info_menu = Menu {
            title: "System Information".to_string(),
            items: vec![
                MenuItem {
                    title: "CPU Information".to_string(),
                    kind: MenuItemKind::Action(|_| {}), // Would show CPU info in a real app
                },
                MenuItem {
                    title: "Memory Information".to_string(),
                    kind: MenuItemKind::Action(|_| {}), // Would show memory info in a real app
                },
                MenuItem {
                    title: "Storage Devices".to_string(),
                    kind: MenuItemKind::Action(|_| {}), // Would show storage info in a real app
                },
                MenuItem {
                    title: "Back".to_string(),
                    kind: MenuItemKind::Action(|app| {
                        app.go_back();
                    }),
                },
            ],
        };
        
        // Boot options submenu
        let boot_menu = Menu {
            title: "Boot Options".to_string(),
            items: vec![
                MenuItem {
                    title: "Boot Priority".to_string(),
                    kind: MenuItemKind::Action(|_| {}), // Would allow changing boot order
                },
                MenuItem {
                    title: "Boot Mode".to_string(),
                    kind: MenuItemKind::Action(|_| {}), // Would allow UEFI/Legacy selection
                },
                MenuItem {
                    title: "Back".to_string(),
                    kind: MenuItemKind::Action(|app| {
                        app.go_back();
                    }),
                },
            ],
        };
        
        // Advanced settings submenu with its own submenu
        let performance_menu = Menu {
            title: "Performance Settings".to_string(),
            items: vec![
                MenuItem {
                    title: "CPU Frequency".to_string(),
                    kind: MenuItemKind::Action(|_| {}), // Would allow CPU frequency adjustment
                },
                MenuItem {
                    title: "CPU Voltage".to_string(),
                    kind: MenuItemKind::Action(|_| {}), // Would allow CPU voltage adjustment
                },
                MenuItem {
                    title: "Memory Frequency".to_string(),
                    kind: MenuItemKind::Action(|_| {}), // Would allow memory frequency adjustment
                },
                MenuItem {
                    title: "Back".to_string(),
                    kind: MenuItemKind::Action(|app| {
                        app.go_back();
                    }),
                },
            ],
        };
        
        let advanced_menu = Menu {
            title: "Advanced Settings".to_string(),
            items: vec![
                MenuItem {
                    title: "Performance Settings".to_string(),
                    kind: MenuItemKind::Submenu(performance_menu),
                },
                MenuItem {
                    title: "Hardware Monitor".to_string(),
                    kind: MenuItemKind::Action(|_| {}), // Would show temps, fan speeds, etc.
                },
                MenuItem {
                    title: "Back".to_string(),
                    kind: MenuItemKind::Action(|app| {
                        app.go_back();
                    }),
                },
            ],
        };

        // Main menu
        let main_menu = Menu {
            title: "Main Menu".to_string(),
            items: vec![
                MenuItem {
                    title: "System Information".to_string(),
                    kind: MenuItemKind::Submenu(system_info_menu),
                },
                MenuItem {
                    title: "Boot Options".to_string(),
                    kind: MenuItemKind::Submenu(boot_menu),
                },
                MenuItem {
                    title: "Advanced Settings".to_string(),
                    kind: MenuItemKind::Submenu(advanced_menu),
                },
                MenuItem {
                    title: "Save and Exit".to_string(),
                    kind: MenuItemKind::Action(|app| {
                        // In a real BIOS, this would save settings
                        app.should_quit = true;
                    }),
                },
                MenuItem {
                    title: "Exit Without Saving".to_string(),
                    kind: MenuItemKind::Action(|app| {
                        // In a real BIOS, this would discard settings
                        app.should_quit = true;
                    }),
                },
            ],
        };

        // Set initial mode based on whether we have a directory
        let (initial_mode, initial_input) = match &experiment_dir_option {
            Some(dir) if dir.exists() => (AppMode::SampleSelection, String::new()),
            Some(dir) => (
                AppMode::DirectoryPrompt, 
                dir.to_string_lossy().to_string()
            ),
            None => (AppMode::DirectoryPrompt, String::new()),
        };
        
        // Determine initial samples list and diagnostics log (empty if no directory provided)
        let (samples, log_entries) = match &experiment_dir_option {
            Some(dir) if dir.exists() => Self::scan_for_samples(dir),
            _ => (Vec::new(), Vec::new()), // Empty samples list if no directory
        };
        
        // Initialize app state
        let mut app = Self {
            mode: initial_mode,
            samples,
            sample_selection_state: ListState::default(),
            selected_sample: None,
            menu_stack: vec![main_menu],
            current_menu_state: ListState::default(),
            should_quit: false,
            popup: {
                if show_diagnostics && !log_entries.is_empty() {
                    PopupState::DiagnosticsLog {
                        log_entries,
                    }
                } else {
                    let input_len = initial_input.len();
                    PopupState::DirectoryInput {
                        input: initial_input,
                        cursor_position: input_len,
                        message: "Enter the experiment directory path:".to_string(),
                    }
                }
            },
            experiment_dir: experiment_dir_option,
            show_diagnostics,
        };
        
        // Select the first sample by default if available
        if !app.samples.is_empty() {
            app.sample_selection_state.select(Some(0));
        }
        
        app
    }
    
    // Scan the experiment directory for samples that have pipeline_summary.json files
    fn scan_for_samples(experiment_dir: &Path) -> (Vec<PathBuf>, Vec<String>) {
        let mut samples = Vec::new();
        let mut log_entries = Vec::new();

        // Check if the experiment directory exists
        if !experiment_dir.exists() {
            log_entries.push(format!("Experiment directory does not exist: {}", experiment_dir.display()));
            return (samples, log_entries);
        }

        // Add diagnostics to log
        log_entries.push(format!("Scanning experiment directory: {}", experiment_dir.display()));
        log_entries.push(String::from("----------------------------------------"));

        // Log directory metadata
        if let Ok(metadata) = std::fs::metadata(experiment_dir) {
            log_entries.push(format!("Directory exists: {}", metadata.is_dir()));
            log_entries.push(format!("Directory permissions: readonly={}, len={} bytes",
                metadata.permissions().readonly(), metadata.len()));
        } else {
            log_entries.push(format!("Failed to get metadata for directory"));
        }

        // Try to list contents of the directory first to diagnose permission issues
        match std::fs::read_dir(experiment_dir) {
            Ok(_) => log_entries.push(format!("Directory is readable")),
            Err(e) => {
                log_entries.push(format!("Error reading directory: {}", e));
                return (samples, log_entries);
            }
        }

        let mut found_subdirs = false;

        // List all subdirectories in the experiment directory
        if let Ok(entries) = std::fs::read_dir(experiment_dir) {
            // Count total entries for diagnostics
            let mut total_entries = 0;
            let mut dirs_count = 0;
            let mut files_count = 0;

            for entry in entries.filter_map(Result::ok) {
                total_entries += 1;

                let path = entry.path();
                if path.is_dir() {
                    dirs_count += 1;
                    found_subdirs = true;
                    // Get only the directory name for cleaner output
                    let dir_name = path.file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("Unknown");

                    // Check if this directory has a pipeline_summary.json file
                    let json_path = path.join("pipeline_summary.json");
                    if json_path.exists() {
                        log_entries.push(format!("✓ {}: Has pipeline_summary.json", dir_name));
                        // Log some metadata about the JSON file for diagnostics
                        if let Ok(json_meta) = std::fs::metadata(&json_path) {
                            log_entries.push(format!("  - Size: {} bytes", json_meta.len()));
                        }
                        samples.push(path);
                    } else {
                        log_entries.push(format!("✗ {}: No pipeline_summary.json", dir_name));
                        // List contents of this directory to help diagnose issues
                        if let Ok(subentries) = std::fs::read_dir(&path) {
                            let subfiles: Vec<String> = subentries
                                .filter_map(Result::ok)
                                .filter(|e| e.path().is_file())
                                .filter_map(|e| e.file_name().to_str().map(String::from))
                                .collect();

                            if !subfiles.is_empty() {
                                log_entries.push(format!("  - Contains files: {}", subfiles.join(", ")));
                            } else {
                                log_entries.push(format!("  - Directory is empty or contains only subdirectories"));
                            }
                        }
                    }
                } else {
                    files_count += 1;
                }
            }

            // Add summary of directory contents to logs
            log_entries.push(format!("Directory contains {} total entries ({} dirs, {} files)",
                total_entries, dirs_count, files_count));
        }

        if !found_subdirs {
            log_entries.push(String::from("No subdirectories found in experiment directory"));
        }

        // If no samples found, add a mock sample for testing
        if samples.is_empty() && cfg!(debug_assertions) {
            log_entries.push(String::from("No samples found with pipeline_summary.json files"));
            log_entries.push(String::from("Adding mock samples for testing"));
            // For development/testing purposes - create a mock sample
            samples.push(PathBuf::from("/mock/sample1"));
            samples.push(PathBuf::from("/mock/sample2"));
        }

        // Sort samples by name for consistent display
        samples.sort();

        log_entries.push(format!("Found {} sample(s) with pipeline_summary.json files", samples.len()));
        log_entries.push(String::from("----------------------------------------"));

        (samples, log_entries)
    }
    
    // Select the current sample and switch to the main menu mode
    fn select_sample(&mut self) {
        if let Some(selected) = self.sample_selection_state.selected() {
            if selected < self.samples.len() {
                self.selected_sample = Some(self.samples[selected].clone());
                self.mode = AppMode::MainMenu;
                
                // Select the first menu item in the main menu
                self.current_menu_state.select(Some(0));
            }
        }
    }
    
    // Return to the sample selection screen
    fn back_to_sample_selection(&mut self) {
        self.mode = AppMode::SampleSelection;
        self.selected_sample = None;
    }
    
    // Methods for handling directory input popup
    
    // Add a character to the directory input
    fn input_add_char(&mut self, c: char) {
        if let PopupState::DirectoryInput { input, cursor_position, .. } = &mut self.popup {
            input.insert(*cursor_position, c);
            *cursor_position += 1;
        }
    }
    
    // Delete a character from the directory input
    fn input_delete_char(&mut self) {
        if let PopupState::DirectoryInput { input, cursor_position, .. } = &mut self.popup {
            if *cursor_position > 0 {
                *cursor_position -= 1;
                input.remove(*cursor_position);
            }
        }
    }
    
    // Move cursor left in the directory input
    fn input_move_cursor_left(&mut self) {
        if let PopupState::DirectoryInput { cursor_position, .. } = &mut self.popup {
            if *cursor_position > 0 {
                *cursor_position -= 1;
            }
        }
    }
    
    // Move cursor right in the directory input
    fn input_move_cursor_right(&mut self) {
        if let PopupState::DirectoryInput { input, cursor_position, .. } = &mut self.popup {
            if *cursor_position < input.len() {
                *cursor_position += 1;
            }
        }
    }
    
    // Submit the directory input and check if it's valid
    fn submit_directory_input(&mut self) {
        if let PopupState::DirectoryInput { input, .. } = &self.popup {
            let input_str = input.clone();

            // Add diagnostic log entry
            let mut input_log = Vec::new();
            input_log.push(format!("Processing directory input: '{}'", input_str));

            // Expand the path (handle ~ in paths)
            let expanded_path = shellexpand::tilde(&input_str);
            let path = PathBuf::from(expanded_path.as_ref());
            input_log.push(format!("Expanded path: '{}'", path.display()));

            // If diagnostics is enabled, log the input processing
            if self.show_diagnostics {
                for entry in &input_log {
                    println!("{}", entry);
                }
            }

            // Check if the directory exists
            if path.exists() {
                input_log.push(format!("Directory exists, proceeding with scan"));

                // Directory exists, continue to sample selection
                self.experiment_dir = Some(path.clone());
                let (new_samples, mut log_entries) = Self::scan_for_samples(&path);

                // Prepend the input log to scan log
                log_entries.splice(0..0, input_log);
                self.samples = new_samples;

                if !self.samples.is_empty() {
                    self.sample_selection_state.select(Some(0));
                    self.mode = AppMode::SampleSelection;
                    // Only show the diagnostics log if requested
                    if self.show_diagnostics {
                        self.popup = PopupState::DiagnosticsLog {
                            log_entries,
                        };
                    } else {
                        self.popup = PopupState::Hidden;
                    }
                } else {
                    // No samples found, show diagnostics even if not requested
                    // as this is an error condition the user should know about
                    self.mode = AppMode::SampleSelection;
                    self.popup = PopupState::DiagnosticsLog {
                        log_entries,
                    };
                }
            } else {
                input_log.push(format!("Directory does not exist, showing confirmation dialog"));

                // If diagnostics is enabled, log the confirmation step
                if self.show_diagnostics {
                    for entry in &input_log {
                        println!("{}", entry);
                    }
                }

                // Directory doesn't exist, ask if we should create it
                self.popup = PopupState::DirectoryConfirmation {
                    directory: path.clone(),
                    message: format!("Directory '{}' does not exist. Create it?", path.display()),
                };
            }
        }
    }
    
    // Handle confirmation response for directory creation
    fn handle_directory_confirmation(&mut self, confirm: bool) {
        if let PopupState::DirectoryConfirmation { directory, .. } = &self.popup {
            let dir_path = directory.clone();

            if confirm {
                // Add diagnostic log for attempted directory creation
                let mut creation_log = Vec::new();
                creation_log.push(format!("Attempting to create directory: {}", dir_path.display()));

                // Create the directory
                match std::fs::create_dir_all(&dir_path) {
                    Ok(_) => {
                        creation_log.push(format!("Successfully created directory"));

                        // If diagnostics enabled, log the creation
                        if self.show_diagnostics {
                            for entry in &creation_log {
                                println!("{}", entry);
                            }
                        }

                        // Directory created, continue to sample selection
                        self.experiment_dir = Some(dir_path.clone());
                        let (new_samples, mut log_entries) = Self::scan_for_samples(&dir_path);

                        // Prepend the creation log to the scan log
                        log_entries.splice(0..0, creation_log);
                        self.samples = new_samples;

                        if !self.samples.is_empty() {
                            self.sample_selection_state.select(Some(0));
                            self.mode = AppMode::SampleSelection;
                            // Only show diagnostics if requested
                            if self.show_diagnostics {
                                self.popup = PopupState::DiagnosticsLog {
                                    log_entries,
                                };
                            } else {
                                self.popup = PopupState::Hidden;
                            }
                        } else {
                            // No samples found, show diagnostics even if not requested
                            // as this is an error condition the user should know about
                            self.mode = AppMode::SampleSelection;
                            self.popup = PopupState::DiagnosticsLog {
                                log_entries,
                            };
                        }
                    },
                    Err(e) => {
                        // Creation failed, log error and show message
                        creation_log.push(format!("Error creating directory: {}", e));

                        // If diagnostics enabled, log the error
                        if self.show_diagnostics {
                            for entry in &creation_log {
                                println!("{}", entry);
                            }
                        }

                        // Show error in input prompt
                        self.popup = PopupState::DirectoryInput {
                            input: dir_path.to_string_lossy().to_string(),
                            cursor_position: dir_path.to_string_lossy().len(),
                            message: format!("Error creating directory: {}", e),
                        };
                    }
                }
            } else {
                // User declined to create directory, return to input
                self.popup = PopupState::DirectoryInput {
                    input: dir_path.to_string_lossy().to_string(),
                    cursor_position: dir_path.to_string_lossy().len(),
                    message: "Enter the experiment directory path:".to_string(),
                };
            }
        }
    }

    fn current_menu(&self) -> &Menu {
        // The last menu in the stack is the current one
        self.menu_stack.last().unwrap()
    }

    fn handle_enter(&mut self) {
        match self.mode {
            AppMode::DirectoryPrompt => {
                match &self.popup {
                    PopupState::DirectoryInput { .. } => {
                        self.submit_directory_input();
                    }
                    PopupState::DirectoryConfirmation { .. } => {
                        self.handle_directory_confirmation(true); // Confirm directory creation
                    }
                    _ => {}
                }
            }
            AppMode::SampleSelection => {
                self.select_sample();
            }
            AppMode::MainMenu => {
                if let Some(selected) = self.current_menu_state.selected() {
                    match &self.current_menu().items[selected].kind {
                        MenuItemKind::Submenu(submenu) => {
                            // Clone the submenu and push it to the stack
                            let submenu_clone = Menu {
                                title: submenu.title.clone(),
                                items: submenu.items.clone(),
                            };
                            self.menu_stack.push(submenu_clone);
                            
                            // Reset selection for the new menu
                            self.current_menu_state.select(Some(0));
                        }
                        MenuItemKind::Action(action) => {
                            // Execute the action
                            action(self);
                        }
                    }
                }
            }
        }
    }

    fn go_back(&mut self) {
        match self.mode {
            AppMode::DirectoryPrompt => {
                match &self.popup {
                    PopupState::DirectoryConfirmation { .. } => {
                        self.handle_directory_confirmation(false); // Cancel directory creation
                    }
                    PopupState::DiagnosticsLog { .. } => {
                        // Close the diagnostics log and return to directory input
                        self.popup = PopupState::DirectoryInput {
                            input: self.experiment_dir
                                .as_ref()
                                .map(|p| p.to_string_lossy().to_string())
                                .unwrap_or_default(),
                            cursor_position: self.experiment_dir
                                .as_ref()
                                .map(|p| p.to_string_lossy().len())
                                .unwrap_or(0),
                            message: "Enter the experiment directory path:".to_string(),
                        };
                    }
                    _ => {
                        // Pressing Escape in the directory input prompt will quit the app
                        self.should_quit = true;
                    }
                }
            }
            AppMode::SampleSelection => {
                // If we're in the diagnostics log popup, close it first
                if let PopupState::DiagnosticsLog { .. } = &self.popup {
                    self.popup = PopupState::Hidden;
                } else if self.experiment_dir.is_none() {
                    // If we're at the sample selection screen without an experiment dir,
                    // go back to directory prompt
                    self.mode = AppMode::DirectoryPrompt;
                    self.popup = PopupState::DirectoryInput {
                        input: String::new(),
                        cursor_position: 0,
                        message: "Enter the experiment directory path:".to_string(),
                    };
                } else {
                    // Otherwise quit the app
                    self.should_quit = true;
                }
            }
            AppMode::MainMenu => {
                if self.menu_stack.len() > 1 {
                    // Remove the current menu from the stack
                    self.menu_stack.pop();
                    
                    // Reset selection for the previous menu
                    self.current_menu_state.select(Some(0));
                } else {
                    // If we're at the main menu's top level, go back to sample selection
                    self.back_to_sample_selection();
                }
            }
        }
    }
    
    // Handle keyboard character input
    fn handle_char(&mut self, c: char) {
        if self.mode == AppMode::DirectoryPrompt {
            if let PopupState::DirectoryInput { .. } = &self.popup {
                if c.is_ascii_graphic() || c == ' ' || c == '/' || c == '\\' || c == '~' || c == '.' {
                    self.input_add_char(c);
                }
            }
        }
    }

    fn next(&mut self) {
        match self.mode {
            AppMode::DirectoryPrompt => {
                // No navigation in directory prompt
            }
            AppMode::SampleSelection => {
                // Navigate through samples
                if !self.samples.is_empty() {
                    let i = match self.sample_selection_state.selected() {
                        Some(i) => (i + 1) % self.samples.len(),
                        None => 0,
                    };
                    self.sample_selection_state.select(Some(i));
                }
            }
            AppMode::MainMenu => {
                // Navigate through menu items
                let items = &self.current_menu().items;
                if !items.is_empty() {
                    let i = match self.current_menu_state.selected() {
                        Some(i) => (i + 1) % items.len(),
                        None => 0,
                    };
                    self.current_menu_state.select(Some(i));
                }
            }
        }
    }

    fn previous(&mut self) {
        match self.mode {
            AppMode::DirectoryPrompt => {
                // No navigation in directory prompt
            }
            AppMode::SampleSelection => {
                // Navigate through samples
                if !self.samples.is_empty() {
                    let i = match self.sample_selection_state.selected() {
                        Some(i) => {
                            if i == 0 {
                                self.samples.len() - 1
                            } else {
                                i - 1
                            }
                        }
                        None => 0,
                    };
                    self.sample_selection_state.select(Some(i));
                }
            }
            AppMode::MainMenu => {
                // Navigate through menu items
                let items = &self.current_menu().items;
                if !items.is_empty() {
                    let i = match self.current_menu_state.selected() {
                        Some(i) => {
                            if i == 0 {
                                items.len() - 1
                            } else {
                                i - 1
                            }
                        }
                        None => 0,
                    };
                    self.current_menu_state.select(Some(i));
                }
            }
        }
    }
}

use clap::Parser;

/// Command line arguments structure
#[derive(Parser, Debug)]
#[command(author, version, about = "Pipeline interface")]
struct Args {
    /// Path to the experiment directory (if not provided, will prompt for input)
    #[arg(short, long)]
    experiment_dir: Option<String>,

    /// Run in command-line mode without TUI
    #[arg(long, default_value_t = false)]
    no_tui: bool,

    /// Show diagnostic logs (defaults to false if not specified)
    #[arg(long, default_value_t = false)]
    diagnostics: bool,
}

// These functions were removed as they were unused and their functionality
// has been integrated directly into the run_cli_mode function and App::new

fn main() {
    // Parse command-line arguments
    let args = Args::parse();

    // Process the experiment directory path if provided
    let experiment_dir_option = args.experiment_dir.as_ref().map(|path_str| {
        let expanded_path = shellexpand::tilde(path_str);
        PathBuf::from(expanded_path.as_ref())
    });

    // Convert to PathBuf for CLI mode
    let cli_dir_path = experiment_dir_option.clone()
        .map(|p| p)
        .unwrap_or_else(|| PathBuf::from("."));

    // Check if the TUI mode should be avoided (useful when running in environments that don't support TUI)
    if args.no_tui {
        run_cli_mode(cli_dir_path, args.diagnostics);
    } else {
        // Try running in TUI mode, fallback to CLI mode if it fails
        if let Err(err) = run_app(experiment_dir_option, args.diagnostics) {
            // Clean up terminal even if there's an error
            let _ = disable_raw_mode();
            let _ = execute!(stdout(), LeaveAlternateScreen);

            // Print the error
            eprintln!("Error: {}", err);

            // Check if it's a terminal compatibility issue
            if err.to_string().contains("Device not configured") {
                eprintln!("\nThis application requires a compatible terminal. Falling back to command-line mode...\n");
                run_cli_mode(cli_dir_path, args.diagnostics);
            } else {
                std::process::exit(1);
            }
        }
    }
}

// Command-line mode for non-interactive environments
fn run_cli_mode(experiment_dir_option: PathBuf, show_diagnostics: bool) {
    println!("Running in command-line mode");
    
    // Get the experiment directory from input if not provided
    let experiment_dir = if !experiment_dir_option.exists() {
        println!("Experiment directory '{}' does not exist or was not provided.", experiment_dir_option.display());
        println!("Please enter the experiment directory path:");
        
        let mut input = String::new();
        stdin().read_line(&mut input).expect("Failed to read line");
        let input = input.trim();
        
        let expanded_path = shellexpand::tilde(input);
        let dir_path = PathBuf::from(expanded_path.as_ref());
        
        if !dir_path.exists() {
            println!("Directory '{}' does not exist. Would you like to create it? (y/n)", dir_path.display());
            
            let mut confirm = String::new();
            stdin().read_line(&mut confirm).expect("Failed to read line");
            
            if confirm.trim().to_lowercase() == "y" {
                match std::fs::create_dir_all(&dir_path) {
                    Ok(_) => {
                        println!("Created directory: {}", dir_path.display());
                        dir_path
                    }
                    Err(e) => {
                        println!("Error creating directory: {}", e);
                        println!("Using current directory instead.");
                        PathBuf::from(".")
                    }
                }
            } else {
                println!("Using current directory instead.");
                PathBuf::from(".")
            }
        } else {
            dir_path
        }
    } else {
        experiment_dir_option
    };
    
    println!("Experiment directory: {}", experiment_dir.display());
    
    // Scan for samples
    let (samples, log_entries) = App::scan_for_samples(&experiment_dir);

    // Print diagnostic logs only if requested
    if show_diagnostics {
        println!("\nDiagnostic logs:");
        for entry in log_entries {
            println!("{}", entry);
        }
    }
    
    if samples.is_empty() {
        println!("No samples found in the experiment directory.");
        return;
    }
    
    println!("\nAvailable samples:");
    for (i, sample) in samples.iter().enumerate() {
        let sample_name = sample.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("Unknown Sample");
        
        println!("{}. {}", i + 1, sample_name);
    }
    
    println!("\nIn the TUI mode, you would select a sample and then access the pipeline interface menu.");
    println!("This command-line version is provided as a fallback for environments that don't support TUI applications.");
}

fn run_app(experiment_dir_option: Option<PathBuf>, show_diagnostics: bool) -> Result<(), Box<dyn Error>> {

    // Print starting message
    println!("Starting pipeline interface...");
    if let Some(dir) = &experiment_dir_option {
        println!("Experiment directory: {}", dir.display());
    } else {
        println!("No experiment directory provided. Will prompt for one.");
    }

    if show_diagnostics {
        println!("Diagnostic logs enabled.");
    }
    
    // Setup panic hook for proper terminal cleanup
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic| {
        // Restore terminal to normal state
        let _ = disable_raw_mode();
        let _ = execute!(stdout(), LeaveAlternateScreen);

        // Call the original panic hook
        hook(panic);
    }));
    
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state with the experiment directory option and diagnostics flag
    let mut app = App::new(experiment_dir_option, show_diagnostics);

    // Main loop
    while !app.should_quit {
        // Draw the UI
        terminal.draw(|frame| ui(frame, &mut app))?;

        // Handle input events with a timeout to avoid hogging CPU
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => {
                            app.should_quit = true;
                        }
                        KeyCode::Char(c) => {
                            app.handle_char(c);
                            
                            if c == 'j' || c == 'k' {
                                match c {
                                    'j' => app.next(),
                                    'k' => app.previous(),
                                    _ => {}
                                }
                            }
                        }
                        KeyCode::Down => {
                            app.next();
                        }
                        KeyCode::Up => {
                            app.previous();
                        }
                        KeyCode::Left => {
                            if app.mode == AppMode::DirectoryPrompt {
                                app.input_move_cursor_left();
                            } else {
                                app.go_back();
                            }
                        }
                        KeyCode::Right => {
                            if app.mode == AppMode::DirectoryPrompt {
                                app.input_move_cursor_right();
                            } else {
                                app.handle_enter();
                            }
                        }
                        KeyCode::Backspace => {
                            if app.mode == AppMode::DirectoryPrompt {
                                app.input_delete_char();
                            } else {
                                app.go_back();
                            }
                        }
                        KeyCode::Enter => {
                            app.handle_enter();
                        }
                        KeyCode::Esc => {
                            app.go_back();
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}

fn ui(frame: &mut Frame, app: &mut App) {
    // Create the layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Title area
            Constraint::Min(0),    // Menu/sample area
            Constraint::Length(1), // Bottom gap
            Constraint::Length(3), // Controls help
        ])
        .split(frame.size());

    match app.mode {
        AppMode::DirectoryPrompt => {
            // Just render a blank background for the directory prompt
            frame.render_widget(
                Block::default()
                    .borders(Borders::NONE)
                    .style(Style::default().fg(Color::White).bg(Color::Black)),
                frame.size(),
            );
            
            // Render the directory prompt as a popup
            render_directory_prompt(frame, app);
        }
        AppMode::SampleSelection => {
            // Render the sample selection interface
            render_sample_selection(frame, app, &chunks);
            
            // If there's a diagnostics popup, render it on top
            if let PopupState::DiagnosticsLog { .. } = &app.popup {
                render_directory_prompt(frame, app);
            }
        }
        AppMode::MainMenu => {
            // Render the main menu interface
            render_main_menu(frame, app, &chunks);
            
            // If there's a diagnostics popup, render it on top
            if let PopupState::DiagnosticsLog { .. } = &app.popup {
                render_directory_prompt(frame, app);
            }
        }
    }
}

fn render_directory_prompt(frame: &mut Frame, app: &App) {
    let area = frame.size();
    
    match &app.popup {
        PopupState::DirectoryInput { input, cursor_position, message } => {
            // Calculate popup dimensions and position
            let popup_width = area.width.min(60);
            let popup_height = 6;
            let popup_x = (area.width - popup_width) / 2;
            let popup_y = (area.height - popup_height) / 2;
            
            let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);
            
            // Render the input popup
            let popup_block = Block::default()
                .title("Directory Input")
                .borders(Borders::ALL)
                .style(Style::default().fg(Color::White).bg(Color::DarkGray));
            
            // Create layout for the popup content
            let popup_chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([
                    Constraint::Length(1), // Message
                    Constraint::Length(1), // Input field
                    Constraint::Length(1), // Instructions
                ])
                .split(popup_area);
            
            // Render the popup block
            frame.render_widget(popup_block, popup_area);
            
            // Render the message
            let message_paragraph = Paragraph::new(message.clone())
                .style(Style::default().fg(Color::White))
                .alignment(Alignment::Left);
            frame.render_widget(message_paragraph, popup_chunks[0]);
            
            // Render the input field
            let input_text = format!("{} ", input); // Add space for cursor
            let input_paragraph = Paragraph::new(input_text.clone())
                .style(Style::default().fg(Color::White))
                .alignment(Alignment::Left);
            frame.render_widget(input_paragraph, popup_chunks[1]);
            
            // Position the cursor
            if let Some(layout_area) = popup_chunks.get(1) {
                frame.set_cursor(
                    layout_area.x + *cursor_position as u16,
                    layout_area.y,
                );
            }
            
            // Render instructions
            let instructions = "Press Enter to submit, Esc to quit";
            let instructions_paragraph = Paragraph::new(instructions)
                .style(Style::default().fg(Color::Yellow))
                .alignment(Alignment::Center);
            frame.render_widget(instructions_paragraph, popup_chunks[2]);
        }
        PopupState::DirectoryConfirmation { directory, message } => {
            // Calculate popup dimensions and position
            let popup_width = area.width.min(60);
            let popup_height = 6;
            let popup_x = (area.width - popup_width) / 2;
            let popup_y = (area.height - popup_height) / 2;
            
            let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);
            
            // Render the confirmation popup
            let popup_block = Block::default()
                .title("Confirmation")
                .borders(Borders::ALL)
                .style(Style::default().fg(Color::White).bg(Color::DarkGray));
            
            // Create layout for the popup content
            let popup_chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([
                    Constraint::Length(1), // Message
                    Constraint::Length(1), // Directory
                    Constraint::Length(1), // Instructions
                ])
                .split(popup_area);
            
            // Render the popup block
            frame.render_widget(popup_block, popup_area);
            
            // Render the message
            let message_paragraph = Paragraph::new(message.clone())
                .style(Style::default().fg(Color::White))
                .alignment(Alignment::Left);
            frame.render_widget(message_paragraph, popup_chunks[0]);
            
            // Render the directory path
            let dir_text = format!("Path: {}", directory.display());
            let dir_paragraph = Paragraph::new(dir_text)
                .style(Style::default().fg(Color::Green))
                .alignment(Alignment::Left);
            frame.render_widget(dir_paragraph, popup_chunks[1]);
            
            // Render instructions
            let instructions = "Press Enter to confirm, Esc to cancel";
            let instructions_paragraph = Paragraph::new(instructions)
                .style(Style::default().fg(Color::Yellow))
                .alignment(Alignment::Center);
            frame.render_widget(instructions_paragraph, popup_chunks[2]);
        }
        PopupState::DiagnosticsLog { log_entries } => {
            // Add overlay to dim the background
            frame.render_widget(
                Block::default().style(Style::default().bg(Color::Black).fg(Color::Black)),
                area
            );
            
            // Calculate popup dimensions and position - make it larger for logs
            let popup_width = area.width.min(80).max(60);
            let popup_height = area.height.min(20).max(10);
            let popup_x = (area.width - popup_width) / 2;
            let popup_y = (area.height - popup_height) / 2;
            
            let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);
            
            // Render the diagnostics popup
            let popup_block = Block::default()
                .title("Diagnostics Log")
                .borders(Borders::ALL)
                .style(Style::default().fg(Color::White).bg(Color::DarkGray));
            
            // Create layout for the popup content
            frame.render_widget(popup_block.clone(), popup_area);
            let popup_inner_area = popup_block.inner(popup_area);
            
            // Convert log entries to a single text with newlines
            let log_text = if log_entries.is_empty() {
                "No diagnostic information available".to_string()
            } else {
                log_entries.join("\n")
            };
            
            // Render the log content with scrolling
            let log_paragraph = Paragraph::new(log_text)
                .style(Style::default().fg(Color::White))
                .alignment(Alignment::Left)
                .scroll((0, 0))  // Starting scroll position
                .wrap(ratatui::widgets::Wrap { trim: true });
                
            frame.render_widget(log_paragraph, popup_inner_area);
            
            // Render instructions at the bottom
            let instructions = "Press Esc to close diagnostics";
            let instructions_area = Rect::new(
                popup_x, 
                popup_y + popup_height - 1,
                popup_width,
                1
            );
            
            let instructions_paragraph = Paragraph::new(instructions)
                .style(Style::default().fg(Color::Yellow).bg(Color::DarkGray))
                .alignment(Alignment::Center);
            frame.render_widget(instructions_paragraph, instructions_area);
        }
        _ => {}
    }
}

fn render_sample_selection(frame: &mut Frame, app: &mut App, chunks: &[Rect]) {
    // Render the title
    let title_block = Block::default()
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Cyan));
    let title = Paragraph::new("Sample Selection")
        .block(title_block)
        .alignment(Alignment::Center);
    frame.render_widget(title, chunks[0]);

    // Create the sample items
    let items: Vec<ListItem> = if app.samples.is_empty() {
        vec![
            ListItem::new("No samples found. Please check the experiment directory."),
            ListItem::new(""),
            ListItem::new("A valid sample must:"),
            ListItem::new("  1. Be a subdirectory of the experiment directory"),
            ListItem::new("  2. Contain a file named 'pipeline_summary.json'"),
            ListItem::new(""),
            ListItem::new("Press 'Esc' to go back and select a different directory.")
        ]
    } else {
        app.samples
            .iter()
            .map(|path| {
                // Extract only the sample name (directory name) for display
                let sample_name = path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("Unknown Sample");
                
                ListItem::new(sample_name.to_string())
                    .style(Style::default().fg(Color::White))
            })
            .collect()
    };

    // Create a List from the items
    let items_list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Available Samples"))
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ");

    // Render the samples with their state
    frame.render_stateful_widget(items_list, chunks[1], &mut app.sample_selection_state);

    // Render the controls help
    let controls = vec![
        "↑/↓: Navigate",
        "Enter: Select Sample",
        "q: Quit",
    ];
    let controls_text = controls.join(" | ");
    let controls_paragraph = Paragraph::new(controls_text)
        .block(Block::default().borders(Borders::ALL))
        .alignment(Alignment::Center);
    frame.render_widget(controls_paragraph, chunks[3]);
}

fn render_main_menu(frame: &mut Frame, app: &mut App, chunks: &[Rect]) {
    // Get the current menu
    let menu = app.current_menu();

    // Get the selected sample name (if available)
    let sample_name = app.selected_sample
        .as_ref()
        .and_then(|path| path.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or("Unknown Sample");

    // Render the title with the selected sample
    let title_block = Block::default()
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Cyan));
    let title_text = format!("{} - Sample: {}", menu.title, sample_name);
    let title = Paragraph::new(title_text)
        .block(title_block)
        .alignment(Alignment::Center);
    frame.render_widget(title, chunks[0]);

    // Create the menu items
    let items: Vec<ListItem> = menu
        .items
        .iter()
        .map(|item| {
            ListItem::new(item.title.clone())
                .style(Style::default().fg(Color::White))
        })
        .collect();

    // Create a List from the items
    let items_list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Options"))
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ");

    // Render the menu with its state
    frame.render_stateful_widget(items_list, chunks[1], &mut app.current_menu_state);

    // Render the controls help
    let controls = vec![
        "↑/↓: Navigate",
        "Enter: Select",
        "Esc/Backspace: Back",
        "q: Quit",
    ];
    let controls_text = controls.join(" | ");
    let controls_paragraph = Paragraph::new(controls_text)
        .block(Block::default().borders(Borders::ALL))
        .alignment(Alignment::Center);
    frame.render_widget(controls_paragraph, chunks[3]);
}
