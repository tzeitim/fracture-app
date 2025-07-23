from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_widget
import shinyswatch
import polars as pl
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
import pandas as pd
import logging

# Initialize matplotlib
matplotlib.use("agg")

# Use logger that inherits uvicorn's formatting
logger = logging.getLogger("fracture_app")

# Import modules
from modules.config import SYSTEM_PREFIXES, DARK_TEMPLATE, LATTE_TEMPLATE, MOCHA_TEMPLATE
from modules.data_loader import load_database, load_data
from modules.data_processing import get_umis, get_selected_umi_stats, compute_coverage, assemble_umi, sweep_assembly_params
from modules.visualization import (
    format_top_contigs_table, create_graph_plot, create_coverage_distribution_plot, 
    create_reads_per_umi_plot
)
from modules.ui_components import (
    create_data_input_sidebar, create_assembly_controls, 
    create_graph_source_controls, create_graph_controls, 
    create_parameter_sweep_controls, create_theme_controls
)

# Configure polars for pretty printing
pl.Config().set_fmt_str_lengths(666)

# Load database from file
try:
    db = load_database('../parquet_db.txt')
    logger.info(f"Successfully loaded database with {len(db)} entries")
except Exception as e:
    logger.error(f"Error loading database: {e}")
    db = {}

# Define the UI
app_ui = ui.page_fluid(
    ui.tags.style("""
    #shiny-notification-panel {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1050; /* Ensure it appears on top */
    }
    """),
    ui.h2("FRACTURE Explorer"),
    ui.output_ui("top_contigs"),
    ui.output_ui("path_results_output"),

    # Layout with sidebar
    ui.layout_sidebar(
        # Sidebar panel
        ui.sidebar(
            create_data_input_sidebar(db=db),
            width=350,
        ),

        # Main panel
        ui.navset_tab(
            ui.nav_panel("Overview",
                ui.row(
                    ui.column(2,
                        ui.panel_well(
                            ui.h4("Summary Statistics"),
                            ui.p("Total UMIs: ", ui.output_text("total_umis")),
                            ui.p("Median Reads/UMI: ", ui.output_text("median_reads")),
                            ui.p("Read length: ", ui.output_text_verbatim("read_length"))
                        )
                    ),
                    ui.column(6,)
                ),
                ui.row(
                    ui.column(6, 
                        ui.panel_well(
                            output_widget("coverage_dist")  
                        )
                    ),
                    ui.column(6, 
                        ui.panel_well(
                            output_widget("reads_per_umi")
                        )
                    )
                )
            ),
            ui.nav_panel("Graph Explorer",
                ui.row(
                    ui.column(12,
                        ui.h4("Graph Visualization"),
                        ui.panel_well(
                            ui.div(output_widget("unified_graph"), style="min-height: 800px; height: auto; width: 100%")
                        ),
                        ui.hr(),
                        ui.output_text("selection_count"),
                        ui.output_ui("selected_nodes_table")
                    )
                )
            ),
            ui.nav_panel("Assembly Results",
                ui.row(
                    ui.panel_well(
                        ui.h4("Selection UMI Stats"),
                        ui.output_ui("selected_umi_stats")
                    )
                ),
                ui.row(
                    ui.column(3,
                        create_assembly_controls()
                    ),
                    ui.column(4,
                        ui.panel_well(
                            ui.output_ui("assembly_stats"),
                        ),
                    ),
                    ui.column(5,
                        ui.h4("Coverage Plot (from uniqued reads!!)"),
                        ui.output_ui("coverage_plot_container")
                    ),
                ),
            ),
            ui.nav_panel("Sweep Results",
                ui.row(
                    ui.column(2,
                        create_parameter_sweep_controls()
                    ),
                    ui.column(4,
                        ui.panel_well(
                            ui.h4("K-mer vs Coverage Sweep"),
                            ui.div(output_widget("sweep_heatmap"), style="height: 500px;"),
                        )
                    ),
                ),
            ),
            ui.nav_panel("Settings",
                ui.row(
                    ui.column(6,
                        ui.panel_well(
                            ui.h3("Application Settings"),
                            create_theme_controls()
                        )
                    ),
                ),
            ),
            id="main_tabs",
            selected="Assembly Results"  
        )
    ),
    ui.output_ui("theme_css"),
    theme=shinyswatch.theme.slate(),
)

def server(input, output, session):
    # Clean up any temporary files
    os.system("rm -f *__*.dot *__*.csv")
    
    # Debug log for session start
    logger.debug("New user session started - Assembly Results tab will be selected by default")
    
    # Explicitly select the Assembly Results tab on startup
    ui.update_navs("main_tabs", selected="Assembly Results")

    # Reactive data storage
    data = reactive.Value(None)
    assembly_result = reactive.Value(None)
    sweep_results = reactive.Value(None)
    path_results = reactive.Value(None)
    dataset = reactive.Value(None)
    current_template = reactive.Value(DARK_TEMPLATE)
    
    # Node selection state management
    clicked_nodes = reactive.Value(set())  # Store clicked node IDs
    
    # Unified graph state
    current_graph_path = reactive.Value(None)
    graph_source_type = reactive.Value("none")  # "assembly", "upload", or "none"
    
    # Theme handling
    @reactive.Effect
    @reactive.event(input.app_theme)
    def update_theme():
        selected_theme = input.app_theme()
        
        if selected_theme == "latte":
            # Update application theme
            session.send_custom_message("shinyswatch-theme", "minty")
            current_template.set(LATTE_TEMPLATE)
        elif selected_theme == "mocha":
            # Update application theme
            session.send_custom_message("shinyswatch-theme", "darkly")
            current_template.set(MOCHA_TEMPLATE)
        else:
            # Default slate theme
            session.send_custom_message("shinyswatch-theme", "slate")
            current_template.set(DARK_TEMPLATE)
    
    # Update the dataset dropdown when system changes
    @reactive.Effect
    @reactive.event(input.system_prefix)
    def update_dataset_choices():
        if input.input_type() == "remote":
            ui.update_selectize(
                "remote_path",
                choices=db,
                selected=None
            )
    
    @reactive.Effect
    @reactive.event(input.assembly_method)
    def update_graph_type_on_method_change():
        # When assembly method changes, automatically update the graph type
        if input.assembly_method() == "shortest_path":
            ui.update_select("graph_type", selected="preliminary")
        else:
            ui.update_select("graph_type", selected="compressed")
    
    @reactive.Effect
    @reactive.event(input.clear_selection)
    def clear_node_selection():
        # Clear both selection text inputs and clicked nodes
        logger.info("Clear Selection button pressed!")
        ui.update_text("selected_nodes", value="")
        ui.update_text("selected_sequences", value="")
        clicked_nodes.set(set())
        ui.notification_show("Selection cleared!", type="message", duration=2)
        
        # Debug: Log all available inputs to see what selection events exist
        all_inputs = [attr for attr in dir(input) if not attr.startswith('_')]
        selection_inputs = [attr for attr in all_inputs if 'select' in attr.lower() or 'click' in attr.lower()]
        graph_inputs = [attr for attr in all_inputs if 'graph' in attr.lower()]
        logger.info(f"Selection-related inputs: {selection_inputs}")
        logger.info(f"Graph-related inputs: {graph_inputs}")
        
        # Debug: Log available inputs for troubleshooting
        try:
            logger.info(f"Total available inputs: {len(all_inputs)}")
        except Exception as e:
            logger.info(f"Error in debug section: {e}")
    
    # Apply Selection button behavior is handled by the graph reactive events
    
    
    
    @output
    @render.text
    def selection_count():
        """Display the current selection count"""
        try:
            current_nodes = clicked_nodes.get()
            count = len(current_nodes)
            logger.info(f"Selection count function called: {count} nodes - {list(current_nodes)}")
            
            if count == 0:
                return "None selected"
            elif count == 1:
                return f"1 node selected: {list(current_nodes)[0]}"
            else:
                return f"{count} nodes selected: {', '.join(list(current_nodes))}"
        except Exception as e:
            logger.error(f"Error in selection_count: {e}")
            return f"Error: {str(e)}"
    
    @output
    @render.ui
    def selected_nodes_table():
        """Display selected nodes as a table"""
        try:
            current_nodes = clicked_nodes.get()
            if not current_nodes:
                return ui.div(ui.p("No nodes selected", style="font-style: italic; color: #888;"))
            
            # Create a simple table of selected nodes
            table_rows = []
            for i, node_id in enumerate(sorted(current_nodes), 1):
                table_rows.append(
                    ui.div(
                        ui.span(f"{i}. ", style="font-weight: bold; color: #007bff;"),
                        ui.span(node_id, style="font-family: monospace; background-color: #f8f9fa; padding: 2px 6px; border-radius: 3px;"),
                        style="margin: 3px 0; display: flex; align-items: center;"
                    )
                )
            
            return ui.div(
                *table_rows,
                style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #007bff;"
            )
        except Exception as e:
            logger.error(f"Error in selected_nodes_table: {e}")
            return ui.div(ui.p(f"Error: {str(e)}", style="color: red;"))
    
    @output
    @render.ui
    def theme_css():
        theme = input.app_theme()
        if theme == "latte":
            return ui.tags.style("""
            body {
                background-color: #eff1f5 !important;
                color: #4c4f69 !important;
            }
            .well, .panel-well {
                background-color: #e6e9ef !important;
                border-color: #ccd0da !important;
            }
            .nav-tabs {
                border-color: #ccd0da !important;
            }
            .nav-tabs > li.active > a {
                background-color: #e6e9ef !important;
                color: #4c4f69 !important;
                border-color: #ccd0da !important;
            }
            .btn-primary {
                background-color: #1e66f5 !important; /* Catppuccin blue */
                border-color: #1e66f5 !important;
            }
            .selectize-dropdown .active {
                background-color: #bcc0cc !important;
                color: #4c4f69 !important;
            }
            .selectize-control.single .selectize-input {
                background-color: #dce0e8 !important;
                color: #4c4f69 !important;
                border-color: #bcc0cc !important;
            }
            input, select {
                background-color: #dce0e8 !important;
                color: #4c4f69 !important;
                border-color: #bcc0cc !important;
            }
            .radio-inline, .checkbox-inline {
                background-color: #dce0e8 !important;
                color: #4c4f69 !important;
                padding: 5px 10px !important;
                margin: 3px !important;
                border-radius: 4px !important;
                border: 1px solid #bcc0cc !important;
            }
            .radio-inline.active, .checkbox-inline.active {
                background-color: #1e66f5 !important;
                color: white !important;
            }
            a {
                color: #1e66f5 !important; /* Catppuccin blue */
            }
            """)
        elif theme == "mocha":
            return ui.tags.style("""
            body {
                background-color: #1e1e2e !important;
                color: #cdd6f4 !important;
            }
            .well, .panel-well {
                background-color: #181825 !important;
                border-color: #313244 !important;
            }
            .nav-tabs {
                border-color: #313244 !important;
            }
            .nav-tabs > li.active > a {
                background-color: #181825 !important;
                color: #cdd6f4 !important;
                border-color: #313244 !important;
            }
            .btn-primary {
                background-color: #cba6f7 !important; /* Catppuccin purple */
                border-color: #cba6f7 !important;
            }
            .selectize-dropdown .active {
                background-color: #45475a !important;
                color: #cdd6f4 !important;
            }
            .selectize-control.single .selectize-input {
                background-color: #313244 !important;
                color: #cdd6f4 !important;
                border-color: #45475a !important;
            }
            input, select {
                background-color: #313244 !important;
                color: #cdd6f4 !important;
                border-color: #45475a !important;
            }
            .radio-inline, .checkbox-inline {
                background-color: #313244 !important;
                color: #cdd6f4 !important;
                padding: 5px 10px !important;
                margin: 3px !important;
                border-radius: 4px !important;
                border: 1px solid #45475a !important;
            }
            .radio-inline.active, .checkbox-inline.active {
                background-color: #cba6f7 !important;
                color: #1e1e2e !important;
            }
            a {
                color: #b4befe !important; /* Catppuccin lavender */
            }
            """)
        else:
            # Default slate theme styling
            return ui.tags.style("""
            .radio-inline, .checkbox-inline {
                background-color: #444d56 !important;
                color: #ffffff !important;
                padding: 5px 10px !important;
                margin: 3px !important;
                border-radius: 4px !important;
                border: 1px solid #555e67 !important;
            }
            .radio-inline.active, .checkbox-inline.active {
                background-color: #007bff !important;
                color: white !important;
            }
            """)

    @reactive.Effect
    @reactive.event(input.parquet_file_local, input.parquet_file, input.remote_path, input.input_type, input.system_prefix, input.sample_n_umis, input.sample_min_reads, input.sample_umis, input.provided_umi)
    def load_dataset():
        with reactive.isolate():
            data.set(None)
            sweep_results.set(None) 
            assembly_result.set(None)
        ui.update_selectize("umi", choices=[], selected=None)

        try:
            # Determine input type and get file_info
            input_type = input.input_type()
            logger.debug(f"Loading dataset with input_type: {input_type}")
            
            file_info = None
            if input_type == "upload":
                file_info = input.parquet_file()
                logger.debug(f"Upload file info: {file_info}")
            elif input_type == "local":
                file_info = input.parquet_file_local()
                logger.debug(f"Local file path: {file_info}")
            elif input_type == "remote":
                logger.debug(f"Remote path - system: {input.system_prefix()}, path: {input.remote_path()}")
            
            # Load data based on input type
            logger.debug("Calling load_data function...")
            df, file_path_or_error = load_data(
                input_type, 
                file_info,
                system_prefix=input.system_prefix() if input_type == "remote" else None,
                remote_path=input.remote_path() if input_type == "remote" else None,
                sample_umis=input.sample_umis(),
                sample_n_umis=input.sample_n_umis(),
                sample_min_reads=input.sample_min_reads(),
                provided_umi=input.provided_umi()
            )
            
            if df is None:
                logger.error(f"load_data returned None with error: {file_path_or_error}")
                ui.notification_show(file_path_or_error, type='error')
                return
                
            # Successfully loaded data
            logger.info(f"Successfully loaded data with shape: {df.shape}")
            logger.debug(f"Data columns: {df.columns}")
            
            dataset.set(file_path_or_error)
            data.set(df)
            ui.notification_show(f"Successfully loaded data from {file_path_or_error}")
            
            # Update UMI choices
            umis = get_umis(df)
            logger.debug(f"Found {len(umis)} UMIs")
            ui.update_selectize(
                "umi",
                choices=umis,
                selected=umis[0] if umis else None
            )

        except Exception as e:
            import traceback
            error_msg = f"Error loading data: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            with reactive.isolate():
                data.set(None)
                sweep_results.set(None)
                assembly_result.set(None)
            ui.notification_show(error_msg, type="error")

    @output
    @render.text
    def total_umis():
        if data() is None:
            return "No data"
        return str(data().select('umi').n_unique())

    @output
    @render.text 
    def median_reads():
        if data() is None:
            return "No data"
        return str(data().group_by('umi').len().get_column('len').median())

    @output
    @render.text 
    def read_length():
        if data() is None:
            return "No data"
        return str(data().get_column('r2_seq').str.len_chars().describe())

    @output
    @render.text
    def selected_umi_stats():
        if data() is None or not input.umi():
            return "No UMI selected"
        if not input.umi() or input.umi() not in data()['umi'].unique():
            return "No valid UMI selected"
        try:
            umi_reads = get_selected_umi_stats(data(), input.umi())
            theme = input.app_theme()
            text_color = "#4c4f69" if theme == "latte" else "#cdd6f4" if theme == "mocha" else "#ffffff"
            
            return ui.HTML(f"""
            Dataset: <span style="color: {text_color};">{dataset()}</span>
            Reads per UMI: <span style="color: {text_color};">{umi_reads}</span>
            """
                       )
        except Exception as e:
            print(f"Error in selected_umi_stats: {e}")  # Log error
            return "Error calculating UMI stats"
            
    @output
    @render_widget
    def coverage_dist():
        return create_coverage_distribution_plot(data())

    @output
    @render_widget
    def reads_per_umi():
        import plotly.express as px
        return create_reads_per_umi_plot(data())

    @reactive.Effect
    @reactive.event(input.assemble)
    def run_regular_assembly():
        try:
            if data() is None or not input.umi():
                ui.notification_show("Please load data and select a UMI first", type="warning")
                return

            logger.info(f"Starting assembly for UMI: {input.umi()}")
            logger.debug(f"Assembly parameters - method: {input.assembly_method()}, k: {input.kmer_size()}, min_coverage: {input.min_coverage()}")

            # Run assembly
            result = assemble_umi(
                data(),
                target_umi=input.umi(),
                k=int(input.kmer_size()),
                min_coverage=int(input.min_coverage()),
                method=input.assembly_method(),
                auto_k=input.auto_k(),
                export_graphs=True,
                only_largest=True,
                start_anchor=input.start_anchor(),
                min_length=None,
                end_anchor=input.end_anchor(),
            )
            
            logger.info(f"Assembly completed, result shape: {result.shape if result is not None else 'None'}")
            
            # Set the appropriate graph type based on assembly method
            if input.assembly_method() == "shortest_path":
                ui.update_select("graph_type", selected="preliminary")
            else:
                ui.update_select("graph_type", selected="compressed")
                
            handle_assembly_result(result)
            
        except Exception as e:
            import traceback
            error_msg = f"Error in regular assembly: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Traceback: {traceback.format_exc()}")
            ui.notification_show(error_msg, type="error")

    def handle_assembly_result(result):
        """Common function to handle assembly results"""
        try:
            if len(result) > 0:
                contig = result.get_column('contig')[0]
                assembly_result.set(contig)

                # Read path results if they exist
                if input.assembly_method() == "shortest_path":
                    base_path = f"{Path(__file__).parent}/{input.umi()}"
                    path_file = f"{base_path}__path.csv"

                    if Path(path_file).exists():
                        logger.debug(f'Loading path file {path_file}')
                        path_df = (pl.read_csv(path_file)
                                   .with_columns(pl.col('sequence').str.len_chars().alias('length'))
                                   .sort('coverage', 'length', descending=True))

                        path_results.set({
                            'results':result,
                            'path_nodes': path_df['sequence'].to_list(),
                            })
                    else:
                        path_results.set(None)

                ui.notification_show(
                        f"Assembly completed successfully! Contig length: {len(contig)}",
                        type="message"
                        )
            else:
                ui.notification_show("Assembly produced no contigs", type="warning")
                assembly_result.set(None)
                path_results.set(None)
        except Exception as e:
            ui.notification_show(f"Error handling assembly result: {str(e)}", type="error")

    @output
    @render.text
    @reactive.event(assembly_result)  
    def path_results_output():
        if path_results() is None:
            return ""
        
        head_n_paths = 5 

        df = (
                path_results()['results']
                .with_columns(pl.col('contig').str.len_chars().alias('Length'))
                .with_columns(
                    coverage=pl.lit(100), # change the average path weight or similar
                    note_type=pl.lit('_terminal'),
                    outgoing_nodes=pl.lit(100),
                    outgoing_directions=pl.lit('Left'),
                    )
                .rename({'umi':'node_id', 'contig':'sequence'})
                .head(head_n_paths)
                )

        if df.shape[0] == 0:
            return "No path results available"

        pl.Config().set_tbl_width_chars(df.get_column('Length').max()+1)
        return ui.HTML(format_top_contigs_table(df))

    @output
    @render.text
    def assembly_stats():
        if data() is None or assembly_result() is None:
            return "No assembly results available"

        read_count = data().filter(pl.col('umi') == input.umi()).height
        theme = input.app_theme()
        highlight_color = "#d20f39" if theme == "latte" else "#f38ba8" if theme == "mocha" else "#ff8080"  # Red
        param_color = "#fe640b" if theme == "latte" else "#fab387" if theme == "mocha" else "#ffa07a"     # Orange/Peach
        text_color = "#4c4f69" if theme == "latte" else "#cdd6f4" if theme == "mocha" else "#ffffff"     # Text
        
        return ui.HTML(f"""
    Target UMI: <h4>{input.umi()}</h4>
    Assembly Method: {input.assembly_method()}
    Contig Length: <span style="color: {highlight_color};">{len(assembly_result())}</span> 
    Input Reads: <span style="color: {text_color};">{read_count}</span> 
    K-mer Size: <span style="color: {param_color};">{input.kmer_size()}</span> 
    Min Coverage: <span style="color: {param_color};">{input.min_coverage()}</span>"""
                       )

    @output
    @render.text 
    @reactive.event(assembly_result)  
    def top_contigs():
        if data() is None or assembly_result() is None:
            return "No contigs available"

        try:
            base_path = f"{Path(__file__).parent}/{input.umi()}"
            graph_path = f"{base_path}__compressed.csv"

            if not Path(graph_path).exists():
                return "Contig graph data not found"

            df = (
                    pl.read_csv(graph_path)
                    .with_columns(pl.col('sequence').str.len_chars().alias('length'))
                    .sort('coverage', 'length', descending=True)
                    .select('node_id', 'sequence', 'coverage', 'length')
                    .head(5)
                    )

            pl.Config().set_tbl_width_chars(df.get_column('length').max()+1)
            pl.Config().set_tbl_width_chars(input.insert_size())

            return ui.HTML(format_top_contigs_table(df))

        except Exception as e:
            logger.error(f"Error loading top contigs: {str(e)}")
            return "Error loading contig data"

    @reactive.Effect
    @reactive.event(input.run_sweep)
    def run_parameter_sweep():
        if data() is None or not input.umi():
            ui.notification_show("Please load data and select a UMI first", type="warning")
            return

        try:
            k_start = int(input.k_start())
            k_end = int(input.k_end())
            if k_end <= k_start:
                ui.notification_show("K-end must be greater than K-start", type="error")
                return

            cov_start = int(input.cov_start())
            cov_end = int(input.cov_end())
            if cov_end <= cov_start:
                ui.notification_show("Coverage end must be greater than start", type="error")
                return

            ui.notification_show(
                "Running parameter sweep...", 
                type="message"
            )

            logger.info(f"Starting parameter sweep for UMI: {input.umi()}")
            logger.debug(f"K-mer range: {k_start}-{k_end}, step={input.k_step()}")
            logger.debug(f"Coverage range: {cov_start}-{cov_end}")

            result = sweep_assembly_params(
                data(),
                target_umi=input.umi(),
                start_anchor=input.start_anchor(),
                end_anchor=input.end_anchor(),
                k_start=k_start,
                k_end=k_end,
                k_step=int(input.k_step()),
                cov_start=cov_start,
                cov_end=cov_end,
                cov_step=1,
                method=input.assembly_method(),
                min_length=None,
                export_graphs=False,
                prefix=f"{input.umi()}_"
            )

            result_pd = result.to_pandas().set_index('k')
            sweep_results.set(result_pd)
            ui.notification_show(
                "Parameter sweep completed successfully!", 
                type="message"
            )

        except Exception as e:
            logger.error(f"Parameter sweep error: {str(e)}")
            import traceback
            logger.debug(f"Sweep traceback: {traceback.format_exc()}")
            ui.notification_show(
                f"Error during parameter sweep: {str(e)}", 
                type="error"
            )

            # Reset sweep results
            sweep_results.set(None)

    @output
    @render.ui
    @reactive.event(input.enable_coverage_plot)
    def coverage_plot_container():
        """Container that shows coverage plot only if enabled"""
        if not input.enable_coverage_plot():
            return ui.div(
                ui.p("Coverage plot is disabled. Enable it in the Assembly Controls to view coverage."),
                style="height: 500px; display: flex; align-items: center; justify-content: center; font-style: italic; color: #888;"
            )
        else:
            return ui.div(output_widget("coverage_plot"), style="height: 500px;")

    @output
    @render_widget
    @reactive.event(input.umi, data, assembly_result, input.app_theme, input.enable_coverage_plot, input.reference_sequence)
    def coverage_plot():
        # Only generate plot if enabled
        if not input.enable_coverage_plot():
            return go.Figure()
            
        if data() is None or not input.umi() or input.umi() not in data()['umi']:
            empty_fig = go.Figure(layout=current_template()['layout'])
            empty_fig.update_layout(
                    height=500,
                    autosize=True,
                    annotations=[dict(
                        text="No data available",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(color='#ffffff', size=14)
                        )]
                    )
            return empty_fig

        try:
            logger.debug(f"Loading reference sequence for coverage plot (UMI: {input.umi()})")
            
            # Use provided reference sequence if available
            if input.reference_sequence() and input.reference_sequence().strip():
                ref_str = input.reference_sequence().strip()
                logger.debug(f"Using provided reference sequence, length: {len(ref_str)}")
            else:
                # Load reference sequence from file
                mods_path = f"{Path(__file__).parent}/mods.parquet"
                logger.debug(f"Looking for reference file at: {mods_path}")
                
                if not Path(mods_path).exists():
                    raise FileNotFoundError(f"Reference file not found: {mods_path}. Please provide a reference sequence manually.")
                    
                mods = pl.read_parquet(mods_path)
                logger.debug(f"Loaded mods.parquet with {len(mods)} rows")
                
                # Check available modifications
                available_mods = mods['mod'].unique().to_list()
                logger.debug(f"Available modifications: {available_mods}")
                
                ref_rows = mods.filter(pl.col('mod') == 'mod_0')
                if len(ref_rows) == 0:
                    raise ValueError(f"No 'mod_0' reference found. Available modifications: {available_mods}. Please provide a reference sequence manually.")
                    
                ref_str = ref_rows['seq'][0]
                logger.debug(f"Using default reference sequence, length: {len(ref_str)}")
            
            # Validate reference sequence
            if not ref_str or len(ref_str) < 10:
                raise ValueError(f"Invalid reference sequence: too short (length: {len(ref_str)})")
            
            # Compute coverage with error handling
            logger.debug(f"Computing coverage for UMI {input.umi()}")
            result = compute_coverage(data(), input.umi(), ref_str)
            logger.debug(f"Coverage computation completed, result shape: {result.shape}")
            
            # Create plot
            fig = px.line(
                    result,
                    x='covered_positions',
                    y='coverage',
                    labels=dict(
                        x="Position in reference",
                        y="Reads",
                        ),
                    title=f"Coverage of raw reads (ref length: {len(ref_str)})",
                    height=500,
                    )

            fig.update_layout(
                    **current_template()['layout'],
                    xaxis_title="Position",
                    yaxis_title="Coverage",
                    autosize=True,
                    )

            # Ensure axis labels are visible
            fig.update_xaxes(showticklabels=True, title_standoff=25)
            fig.update_yaxes(showticklabels=True, title_standoff=25)

            logger.debug("Coverage plot created successfully")
            return fig
            
        except FileNotFoundError as e:
            error_msg = f"Reference file error: {str(e)}"
            logger.error(error_msg)
            empty_fig = go.Figure(layout=current_template()['layout'])
            empty_fig.update_layout(
                height=500,
                autosize=True,
                annotations=[dict(
                    text=error_msg,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color='#ff6b6b', size=14)
                )]
            )
            return empty_fig
            
        except ValueError as e:
            error_msg = f"Reference data error: {str(e)}"
            logger.error(error_msg)
            empty_fig = go.Figure(layout=current_template()['layout'])
            empty_fig.update_layout(
                height=500,
                autosize=True,
                annotations=[dict(
                    text=error_msg,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color='#ff6b6b', size=14)
                )]
            )
            return empty_fig
            
        except Exception as e:
            import traceback
            error_msg = f"Coverage plot error: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            empty_fig = go.Figure(layout=current_template()['layout'])
            empty_fig.update_layout(
                height=500,
                autosize=True,
                annotations=[dict(
                    text=f"Error: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color='#ff6b6b', size=14)
                )]
            )
            return empty_fig

    @output
    @render_widget
    @reactive.event(sweep_results, input.app_theme)
    def sweep_heatmap():
        if sweep_results() is None:
            # Create empty figure with specific dimensions
            empty_fig = go.Figure(layout=current_template()['layout'])
            empty_fig.update_layout(
                    height=500, 
                    autosize=True,
                    # Optional: Add a message indicating no data
                    annotations=[dict(
                        text="No sweep results available",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(color='#ffffff', size=14)
                        )]
                    )
            return empty_fig

        import plotly.express as px
        fig = px.imshow(
                sweep_results(),
                labels=dict(
                    x="Minimum Coverage",
                    y="K-mer Size",
                    color="Contig Length"
                    ),
                title="K-mer Size vs Coverage Parameter Sweep",
                height=500,
                color_continuous_scale="RdYlBu_r",
                zmax=input.insert_size()*1.05,
                )

        fig.update_layout(
                autosize=True,
                **current_template()['layout'],
                xaxis_title="Minimum Coverage",
                yaxis_title="K-mer Size",
                coloraxis_colorbar_title="Contig Length"
                )

        # Ensure axis labels are visible
        fig.update_xaxes(showticklabels=True, title_standoff=25)
        fig.update_yaxes(showticklabels=True, title_standoff=25)

        return fig

    # Handle assembly graph generation
    @reactive.Effect
    @reactive.event(input.generate_graph)
    def generate_assembly_graph():
        """Generate graph from assembly when button clicked"""
        if assembly_result() is None:
            ui.notification_show("Please run assembly first", type="warning")
            return
        
        # Build path based on selected graph type
        base_path = f"{Path(__file__).parent}/"
        graph_suffix = input.umi()
        graph_path = f"{base_path}{graph_suffix}__{input.graph_type()}.dot"
        
        if Path(graph_path).exists():
            current_graph_path.set(graph_path)
            graph_source_type.set("assembly")
            ui.notification_show(f"Loaded {input.graph_type()} graph", type="success", duration=2)
        else:
            ui.notification_show(f"Graph file not found: {graph_path}", type="error")

    # Handle DOT file upload
    @reactive.Effect
    @reactive.event(input.dot_file)
    def handle_dot_upload():
        """Handle uploaded DOT file"""
        if input.dot_file() is not None:
            file_info = input.dot_file()
            dot_file_path = file_info[0]["datapath"]
            logger.info(f"DOT file uploaded: {dot_file_path}")
            
            # Set the radio button to upload mode
            ui.update_radio_buttons("graph_source", selected="upload")
            
            current_graph_path.set(dot_file_path)
            graph_source_type.set("upload")
            ui.notification_show("DOT file loaded", type="success", duration=2)
        else:
            logger.info("DOT file cleared or None")

    # Unified graph rendering function
    @output
    @render_widget
    @reactive.event(
        current_graph_path, 
        input.app_theme,
        input.use_weighted,
        input.weight_method,
        input.separate_components,
        input.component_padding,
        input.min_component_size,
        input.layout_k,
        input.layout_iterations,
        input.layout_scale,
        input.selected_nodes,
        input.selected_sequences,
        clicked_nodes
    )
    def unified_graph():
        """Render the unified graph widget"""
        logger.info(f"Rendering unified graph - source: {graph_source_type.get()}, path: {current_graph_path.get()}")
        
        # Check if we have a graph to display
        graph_path = current_graph_path.get()
        if graph_path is None:
            empty_fig = go.FigureWidget(layout=current_template()['layout'])
            empty_fig.update_layout(
                height=800,
                autosize=True,
                annotations=[dict(
                    text="Generate a graph from assembly or upload a DOT file",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(color='#888', size=16)
                )]
            )
            return empty_fig
        
        try:
            # Parse selected nodes and sequences
            selected_nodes = None
            selected_sequences = None
            
            # Get all selected nodes (from clicks and text input)
            all_selected_nodes = set(clicked_nodes.get())
            
            if input.selected_nodes() and input.selected_nodes().strip():
                text_nodes = [node.strip() for node in input.selected_nodes().split(',') if node.strip()]
                all_selected_nodes.update(text_nodes)
            
            if all_selected_nodes:
                selected_nodes = list(all_selected_nodes)
                logger.debug(f"Selected nodes: {selected_nodes}")
            
            if input.selected_sequences() and input.selected_sequences().strip():
                selected_sequences = [seq.strip() for seq in input.selected_sequences().split(',') if seq.strip()]
            
            # Determine path nodes if this is from assembly
            path_nodes = None
            if graph_source_type.get() == "assembly" and path_results() is not None:
                path_data = path_results()
                if isinstance(path_data, list) and len(path_data) > 0:
                    path_nodes = set()
                    for path in path_data:
                        if 'path' in path:
                            path_nodes.update(path['path'])
            
            # Check if file exists
            if not Path(graph_path).exists():
                logger.warning(f"Graph file not found: {graph_path}")
                error_fig = go.FigureWidget(layout=current_template()['layout'])
                error_fig.update_layout(
                    height=800,
                    annotations=[dict(
                        text=f"Graph file not found: {graph_path}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(color='#ff0000', size=14)
                    )]
                )
                return error_fig
            
            logger.info(f"Creating graph from file: {graph_path}")
            
            # Create the graph plot  
            fig = create_graph_plot(
                graph_path,
                dark_mode=(input.app_theme() != "latte"),
                weighted=input.use_weighted(),
                weight_method=input.weight_method(),
                separate_components=input.separate_components(),
                component_padding=input.component_padding(),
                min_component_size=input.min_component_size(),
                selected_nodes=selected_nodes,
                selected_sequences=selected_sequences,
                path_nodes=path_nodes,
                spring_args={
                    'k': input.layout_k(),
                    'iterations': input.layout_iterations(),
                    'scale': input.layout_scale()
                }
            )
            
            # Convert to FigureWidget for interactivity
            fig_widget = go.FigureWidget(fig)
            
            # Add click handler
            def on_node_click(trace, points, selector):
                if not points.point_inds:
                    return
                
                point_ind = points.point_inds[0]
                
                if hasattr(trace, 'customdata') and trace.customdata is not None:
                    node_id = str(trace.customdata[point_ind])
                    logger.info(f"Graph node clicked: {node_id}")
                    
                    # Toggle selection
                    current_selection = clicked_nodes.get().copy()
                    
                    if node_id in current_selection:
                        current_selection.discard(node_id)
                        ui.notification_show(f"Deselected: {node_id}", type="message", duration=1)
                    else:
                        current_selection.add(node_id)
                        ui.notification_show(f"Selected: {node_id}", type="message", duration=1)
                    
                    clicked_nodes.set(current_selection)
                    
                    # Update text input
                    if current_selection:
                        ui.update_text("selected_nodes", value=", ".join(sorted(current_selection)))
                    else:
                        ui.update_text("selected_nodes", value="")
            
            # Attach click handler to nodes trace
            for trace in fig_widget.data:
                if hasattr(trace, 'name') and trace.name == 'nodes':
                    trace.on_click(on_node_click)
                    break
            
            return fig_widget
            
        except Exception as e:
            logger.error(f"Error creating graph: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            error_fig = go.FigureWidget(layout=current_template()['layout'])
            error_fig.update_layout(
                height=800,
                annotations=[dict(
                    text=f"Error: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(color='#ff0000', size=14)
                )]
            )
            return error_fig

app = App(app_ui, server)
