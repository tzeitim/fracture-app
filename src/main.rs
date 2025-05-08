use std::error::Error;
use std::io::stdout;
use std::time::Duration;

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

// Define our app state
struct App {
    // Navigation state
    menu_stack: Vec<Menu>,
    current_menu_state: ListState,
    // Whether the app should exit
    should_quit: bool,
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
    fn new() -> Self {
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

        // Initialize with main menu
        let mut app = Self {
            menu_stack: vec![main_menu],
            current_menu_state: ListState::default(),
            should_quit: false,
        };
        
        // Select the first item by default
        app.current_menu_state.select(Some(0));
        
        app
    }

    fn current_menu(&self) -> &Menu {
        // The last menu in the stack is the current one
        self.menu_stack.last().unwrap()
    }

    fn handle_enter(&mut self) {
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

    fn go_back(&mut self) {
        if self.menu_stack.len() > 1 {
            // Remove the current menu from the stack
            self.menu_stack.pop();
            
            // Reset selection for the previous menu
            self.current_menu_state.select(Some(0));
        }
    }

    fn next(&mut self) {
        let items = &self.current_menu().items;
        if !items.is_empty() {
            let i = match self.current_menu_state.selected() {
                Some(i) => (i + 1) % items.len(),
                None => 0,
            };
            self.current_menu_state.select(Some(i));
        }
    }

    fn previous(&mut self) {
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

fn main() -> Result<(), Box<dyn Error>> {
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

    // Create app state
    let mut app = App::new();

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
                        KeyCode::Char('j') | KeyCode::Down => {
                            app.next();
                        }
                        KeyCode::Char('k') | KeyCode::Up => {
                            app.previous();
                        }
                        KeyCode::Char('h') | KeyCode::Left | KeyCode::Backspace | KeyCode::Esc => {
                            app.go_back();
                        }
                        KeyCode::Char('l') | KeyCode::Right | KeyCode::Enter => {
                            app.handle_enter();
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
            Constraint::Min(0),    // Menu area
            Constraint::Length(1), // Bottom gap
            Constraint::Length(3), // Controls help
        ])
        .split(frame.size());

    // Get the current menu
    let menu = app.current_menu();

    // Render the title
    let title_block = Block::default()
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Cyan));
    let title = Paragraph::new(menu.title.clone())
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
