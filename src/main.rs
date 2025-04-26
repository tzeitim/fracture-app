use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Duration;
use std::time::Instant;

use color_eyre::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use polars::prelude::*;
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::fs::File;

/// App holds the state of the application
struct App {
    /// Current state of the application
    state: AppState,
    /// Which tab is currently selected
    current_tab: usize,
    /// Whether the app should exit
    should_quit: bool,
    /// Pipeline status
    pipeline_status: PipelineStatus,
    /// Pipeline summary data
    summary: Option<PipelineSummary>,
    /// Current tab data
    current_data: Option<TabData>,
    /// Directory where pipeline is running
    workdir: String,
    /// Target sample ID
    target_sample: String,
}

/// Possible states the app can be in
enum AppState {
    Normal,
    Loading,
    Error(String),
}

/// Status of the pipeline
#[derive(Debug, Default, Clone)]
struct PipelineStatus {
    current_step: Option<String>,
    completed_steps: Vec<String>,
    running: bool,
    progress: f64,
}

/// Summary of the pipeline run
#[derive(Debug, Default, Clone)]
struct PipelineSummary {
    total_reads: Option<usize>,
    total_umis: Option<usize>,
    success_rate: Option<f64>,
    mean_contig_length: Option<f64>,
    avg_reads_per_umi: Option<f64>,
}

/// Data for the current tab
enum TabData {
    UmiStats(DataFrame),
    Contigs(DataFrame),
    Logs(Vec<String>),
    None,
}

impl Default for App {
    fn default() -> Self {
        Self {
            state: AppState::Normal,
            current_tab: 0,
            should_quit: false,
            pipeline_status: PipelineStatus::default(),
            summary: None,
            current_data: None,
            workdir: String::from(""),
            target_sample: String::from(""),
        }
    }
}

impl App {
    /// Create a new App with the specified workdir and target sample
    fn new(workdir: String, target_sample: String) -> Self {
        let mut app = App::default();
        app.workdir = workdir;
        app.target_sample = target_sample;
        app
    }

    /// Loads data for the current tab
    fn load_data(&mut self) -> Result<(), Box<dyn Error>> {
        self.state = AppState::Loading;
        self.load_summary()?;
        
        match self.current_tab {
            0 => self.load_summary_data()?,
            1 => self.load_umi_stats()?,
            2 => self.load_contigs_data()?,
            3 => self.load_logs()?,
            _ => self.current_data = Some(TabData::None),
        }
        
        self.state = AppState::Normal;
        Ok(())
    }
    
    /// Loads a summary of the pipeline run
    fn load_summary(&mut self) -> Result<(), Box<dyn Error>> {
        // Check if pipeline_summary.json exists
        let summary_path = format!("{}/{}/pipeline_summary.json", self.workdir, self.target_sample);
        if !Path::new(&summary_path).exists() {
            self.summary = Some(PipelineSummary::default());
            return Ok(());
        }
        
        // Read and parse summary
        let summary_content = fs::read_to_string(summary_path)?;
        let summary_json: serde_json::Value = serde_json::from_str(&summary_content)?;
        
        let mut summary = PipelineSummary::default();
        
        // Extract metrics from parquet step if available
        if let Some(parquet) = summary_json.get("parquet") {
            if let Some(metrics) = parquet.get("metrics") {
                if let Some(total_reads) = metrics.get("total_reads") {
                    summary.total_reads = total_reads.as_u64().map(|v| v as usize);
                }
                if let Some(total_umis) = metrics.get("total_umis") {
                    summary.total_umis = total_umis.as_u64().map(|v| v as usize);
                }
            }
        }
        
        // Extract metrics from fracture step if available
        if let Some(fracture) = summary_json.get("fracture") {
            if let Some(metrics) = fracture.get("metrics") {
                if let Some(success_rate) = metrics.get("success_rate") {
                    summary.success_rate = success_rate.as_f64();
                }
                if let Some(mean_length) = metrics.get("mean_contig_length") {
                    summary.mean_contig_length = mean_length.as_f64();
                }
            }
        }
        
        // Collect completed steps
        let mut completed_steps = Vec::new();
        for (step_name, _) in summary_json.as_object().unwrap_or(&serde_json::Map::new()) {
            completed_steps.push(step_name.to_string());
        }
        
        self.pipeline_status.completed_steps = completed_steps;
        self.summary = Some(summary);
        
        Ok(())
    }
    
    /// Load summary data for the dashboard tab
    fn load_summary_data(&mut self) -> Result<(), Box<dyn Error>> {
        // Summary data is already loaded in load_summary()
        self.current_data = Some(TabData::None);
        Ok(())
    }
    
    /// Load UMI statistics data
    fn load_umi_stats(&mut self) -> Result<(), Box<dyn Error>> {
        let parsed_reads_path = format!("{}/{}/parsed_reads.parquet", self.workdir, self.target_sample);
        
        if !Path::new(&parsed_reads_path).exists() {
            self.current_data = Some(TabData::None);
            return Ok(());
        }
        
        // Read parquet file
        let df = ParquetReader::new(File::open(parsed_reads_path)?)
            .finish()?
            .lazy()
            .select([
                col("umi"),
                col("reads"),
            ])
            .group_by([col("umi")])
            .agg([
                len().alias("count"),
            ])
            .sort("count", SortOptions { descending: false, ..Default::default() })
            .limit(100)
            .collect()?;
            
        self.current_data = Some(TabData::UmiStats(df));
        Ok(())
    }
    
    /// Load contigs data
    fn load_contigs_data(&mut self) -> Result<(), Box<dyn Error>> {
        let contigs_path = format!("{}/{}/contigs_pl_false.parquet", self.workdir, self.target_sample);
        
        if !Path::new(&contigs_path).exists() {
            self.current_data = Some(TabData::None);
            return Ok(());
        }
        
        // Read parquet file
        let df = ParquetReader::new(File::open(contigs_path)?)
            .finish()?
            .lazy()
            .select([
                col("umi"),
                col("contig"),
                col("length"),
                col("failed"),
                col("k"),
                col("min_coverage"),
            ])
            .sort("length", SortOptions { descending: true, ..Default::default() })
            .limit(100)
            .collect()?;
            
        self.current_data = Some(TabData::Contigs(df));
        Ok(())
    }
    
    /// Load log data
    fn load_logs(&mut self) -> Result<(), Box<dyn Error>> {
        let log_dir = format!("{}/{}/logs", self.workdir, self.target_sample);
        let log_path = format!("{}/fracture.log", log_dir);
        
        if !Path::new(&log_path).exists() {
            self.current_data = Some(TabData::Logs(vec!["No logs found".to_string()]));
            return Ok(());
        }
        
        // Read log file
        let logs = fs::read_to_string(log_path)?
            .lines()
            .map(String::from)
            .collect();
            
        self.current_data = Some(TabData::Logs(logs));
        Ok(())
    }

    /// Changes to the next tab
    fn next_tab(&mut self) {
        self.current_tab = (self.current_tab + 1) % 4;
        if let Err(e) = self.load_data() {
            self.state = AppState::Error(format!("Error loading data: {}", e));
        }
    }

    /// Changes to the previous tab
    fn previous_tab(&mut self) {
        self.current_tab = (self.current_tab + 3) % 4;
        if let Err(e) = self.load_data() {
            self.state = AppState::Error(format!("Error loading data: {}", e));
        }
    }

    /// Quits the application
    fn quit(&mut self) {
        self.should_quit = true;
    }
}

/// Runs the application
fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    mut app: App,
    tick_rate: Duration,
) -> io::Result<()> {
    let mut last_tick = Instant::now();
    
    // Initial data load
    if let Err(e) = app.load_data() {
        app.state = AppState::Error(format!("Error loading data: {}", e));
    }
    
    loop {
        terminal.draw(|f| ui(f, &app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));
        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => app.quit(),
                        KeyCode::Char('n') | KeyCode::Tab | KeyCode::Right => app.next_tab(),
                        KeyCode::Char('p') | KeyCode::BackTab | KeyCode::Left => app.previous_tab(),
                        KeyCode::Char('r') => {
                            if let Err(e) = app.load_data() {
                                app.state = AppState::Error(format!("Error loading data: {}", e));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        if last_tick.elapsed() >= tick_rate {
            // Periodic refresh
            if let Err(e) = app.load_data() {
                app.state = AppState::Error(format!("Error loading data: {}", e));
            }
            last_tick = Instant::now();
        }
        if app.should_quit {
            return Ok(());
        }
    }
}

/// Renders the user interface
fn ui(f: &mut Frame, app: &App) {
    // Create the layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(f.size());

    // Title and tabs
    let titles = vec!["Dashboard", "UMIs", "Contigs", "Logs"];
    let tabs = Tabs::new(titles)
        .select(app.current_tab)
        .style(Style::default().fg(Color::White))
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        )
        .divider("|");
        
    f.render_widget(tabs, chunks[0]);
    
    // Main content
    match app.current_tab {
        0 => render_dashboard(f, app, chunks[1]),
        1 => render_umis(f, app, chunks[1]),
        2 => render_contigs(f, app, chunks[1]),
        3 => render_logs(f, app, chunks[1]),
        _ => {}
    }
    
    // Footer with help text
    let footer_text = match app.state {
        AppState::Normal => "q: Quit, n/tab: Next tab, p/shift+tab: Previous tab, r: Refresh".to_string(),
        AppState::Loading => "Loading data...".to_string(),
        AppState::Error(ref e) => format!("Error: {}", e),
    };
    
    let footer = Paragraph::new(footer_text)
        .style(Style::default().fg(Color::White))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::TOP));
    
    f.render_widget(footer, chunks[2]);
}

/// Renders the dashboard tab
fn render_dashboard(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Title
            Constraint::Length(5),  // Progress
            Constraint::Length(10), // Summary
            Constraint::Min(0),     // Steps
        ])
        .split(area);
    
    // Title
    let title = Paragraph::new(format!("Sample: {}", app.target_sample))
        .style(Style::default().fg(Color::White))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::NONE));
    
    f.render_widget(title, chunks[0]);
    
    // Progress gauge
    render_progress(f, app, chunks[1]);
    
    // Summary
    render_summary(f, app, chunks[2]);
    
    // Steps
    render_steps(f, app, chunks[3]);
}

/// Renders progress information
fn render_progress(f: &mut Frame, app: &App, area: Rect) {
    let completed_steps = app.pipeline_status.completed_steps.len();
    let total_steps = 4; // parquet, preprocess, fracture, qc
    let progress = (completed_steps as f64) / (total_steps as f64);
    
    let gauge = Gauge::default()
        .block(Block::default().title("Pipeline Progress").borders(Borders::ALL))
        .gauge_style(Style::default().fg(Color::Cyan).bg(Color::Black))
        .percent((progress * 100.0) as u16);
    
    f.render_widget(gauge, area);
}

/// Renders summary information
fn render_summary(f: &mut Frame, app: &App, area: Rect) {
    let summary = app.summary.clone().unwrap_or_default();
    
    let summary_text = vec![
        Line::from(vec![
            Span::styled("Total Reads: ", Style::default().fg(Color::Green)),
            Span::raw(summary.total_reads.map_or("N/A".to_string(), |v| format!("{}", v))),
        ]),
        Line::from(vec![
            Span::styled("Total UMIs: ", Style::default().fg(Color::Green)),
            Span::raw(summary.total_umis.map_or("N/A".to_string(), |v| format!("{}", v))),
        ]),
        Line::from(vec![
            Span::styled("Success Rate: ", Style::default().fg(Color::Green)),
            Span::raw(summary.success_rate.map_or("N/A".to_string(), |v| format!("{:.2}%", v))),
        ]),
        Line::from(vec![
            Span::styled("Mean Contig Length: ", Style::default().fg(Color::Green)),
            Span::raw(summary.mean_contig_length.map_or("N/A".to_string(), |v| format!("{:.2}", v))),
        ]),
    ];
    
    let summary_paragraph = Paragraph::new(summary_text)
        .style(Style::default().fg(Color::White))
        .block(Block::default().title("Summary").borders(Borders::ALL));
    
    f.render_widget(summary_paragraph, area);
}

/// Renders pipeline steps
fn render_steps(f: &mut Frame, app: &App, area: Rect) {
    let steps = vec![
        "parquet", "preprocess", "fracture", "qc"
    ];
    
    let completed_steps = &app.pipeline_status.completed_steps;
    
    let items: Vec<ListItem> = steps
        .iter()
        .map(|&step| {
            let is_completed = completed_steps.contains(&step.to_string());
            let line = if is_completed {
                Line::from(vec![
                    Span::styled("✓ ", Style::default().fg(Color::Green)),
                    Span::styled(step, Style::default().fg(Color::White)),
                ])
            } else {
                Line::from(vec![
                    Span::styled("□ ", Style::default().fg(Color::Gray)),
                    Span::styled(step, Style::default().fg(Color::Gray)),
                ])
            };
            
            ListItem::new(line)
        })
        .collect();
    
    let list = List::new(items)
        .block(Block::default().title("Pipeline Steps").borders(Borders::ALL))
        .highlight_style(Style::default().bg(Color::DarkGray));
    
    f.render_widget(list, area);
}

/// Renders the UMIs tab
fn render_umis(f: &mut Frame, app: &App, area: Rect) {
    match &app.current_data {
        Some(TabData::UmiStats(df)) => {
            // Create a table for UMI stats
            let header = Row::new(vec!["UMI", "Count"])
                .style(Style::default().fg(Color::Yellow));
            
            // Get rows from dataframe
            let rows: Vec<Row> = (0..df.height().min(100)) // Limit to at most 100 rows for performance
                .map(|i| {
                    let umi = df.column("umi").unwrap().get(i).unwrap().to_string();
                    let count = df.column("count").unwrap().get(i).unwrap().to_string();
                    Row::new(vec![umi, count])
                })
                .collect();
                
            let table = Table::new(
                rows,
                [
                    Constraint::Percentage(70),
                    Constraint::Percentage(30),
                ],
            )
            .header(header)
            .block(Block::default().title("UMI Statistics").borders(Borders::ALL))
            .highlight_style(Style::default().add_modifier(Modifier::REVERSED))
            .highlight_symbol("> ");
            
            f.render_widget(table, area);
        },
        _ => {
            let message = if matches!(app.state, AppState::Loading) {
                "Loading UMI data..."
            } else {
                "No UMI data available"
            };
            
            let paragraph = Paragraph::new(message)
                .style(Style::default().fg(Color::White))
                .alignment(Alignment::Center)
                .block(Block::default().title("UMI Statistics").borders(Borders::ALL));
                
            f.render_widget(paragraph, area);
        }
    }
}

/// Renders the contigs tab
fn render_contigs(f: &mut Frame, app: &App, area: Rect) {
    match &app.current_data {
        Some(TabData::Contigs(df)) => {
            // Create a table for contig data
            let header = Row::new(vec!["UMI", "Length", "K", "Min Coverage", "Status"])
                .style(Style::default().fg(Color::Yellow));
            
            // Get rows from dataframe
            let rows: Vec<Row> = (0..df.height().min(100)) // Limit to at most 100 rows for performance
                .map(|i| {
                    let umi = df.column("umi").unwrap().get(i).unwrap().to_string();
                    let length = df.column("length").unwrap().get(i).unwrap().to_string();
                    let k = df.column("k").unwrap().get(i).unwrap().to_string();
                    let min_coverage = df.column("min_coverage").unwrap().get(i).unwrap().to_string();
                    let failed = df.column("failed").unwrap().get(i).unwrap().to_string();
                    
                    let status = if failed == "true" { "Failed" } else { "Success" };
                    let status_style = if status == "Failed" { 
                        Style::default().fg(Color::Red) 
                    } else { 
                        Style::default().fg(Color::Green) 
                    };
                    
                    Row::new(vec![
                        umi,
                        length,
                        k,
                        min_coverage,
                        status.to_string(),
                    ]).style(status_style)
                })
                .collect();
                
            let table = Table::new(
                rows,
                [
                    Constraint::Percentage(30),
                    Constraint::Percentage(15),
                    Constraint::Percentage(15),
                    Constraint::Percentage(20),
                    Constraint::Percentage(20),
                ],
            )
            .header(header)
            .block(Block::default().title("Contigs").borders(Borders::ALL))
            .highlight_style(Style::default().add_modifier(Modifier::REVERSED))
            .highlight_symbol("> ");
            
            f.render_widget(table, area);
        },
        _ => {
            let message = if matches!(app.state, AppState::Loading) {
                "Loading contig data..."
            } else {
                "No contig data available"
            };
            
            let paragraph = Paragraph::new(message)
                .style(Style::default().fg(Color::White))
                .alignment(Alignment::Center)
                .block(Block::default().title("Contigs").borders(Borders::ALL));
                
            f.render_widget(paragraph, area);
        }
    }
}

/// Renders the logs tab
fn render_logs(f: &mut Frame, app: &App, area: Rect) {
    let default_log = if matches!(app.state, AppState::Loading) {
        vec!["Loading logs...".to_string()]
    } else {
        vec!["No logs available".to_string()]
    };
    
    let logs = match &app.current_data {
        Some(TabData::Logs(logs)) => logs,
        _ => &default_log
    };
    
    let log_items: Vec<ListItem> = logs
        .iter()
        .map(|log| {
            let style = if log.contains("ERROR") {
                Style::default().fg(Color::Red)
            } else if log.contains("WARNING") {
                Style::default().fg(Color::Yellow)
            } else if log.contains("INFO") {
                Style::default().fg(Color::Blue)
            } else {
                Style::default()
            };
            
            ListItem::new(log.clone()).style(style)
        })
        .collect();
    
    let logs_list = List::new(log_items)
        .block(Block::default().title("Logs").borders(Borders::ALL))
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED))
        .highlight_symbol("> ");
    
    f.render_widget(logs_list, area);
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    
    let (workdir, target_sample) = if args.len() >= 3 {
        (args[1].clone(), args[2].clone())
    } else {
        eprintln!("Usage: fracture-tui <workdir> <target_sample>");
        eprintln!("Example: fracture-tui /path/to/workdir sample_id");
        return Ok(());
    };
    
    // Create app
    let app = App::new(workdir, target_sample);
    
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    
    // Run the app
    let tick_rate = Duration::from_millis(500);
    let result = run_app(&mut terminal, app, tick_rate);
    
    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    
    if let Err(err) = result {
        println!("Error: {}", err);
    }
    
    Ok(())
}
