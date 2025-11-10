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

app_dir = Path(__file__).parent
www_dir = app_dir / "www"
os.environ["SHINY_MOUNT_DIRECTORIES"] = f"/:{www_dir}"

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




os.environ['NUMEXPR_MAX_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

# Configure polars for pretty printing
pl.Config().set_fmt_str_lengths(666)

# Read configuration from environment variables (set by run.app)
# These can be customized via command-line arguments to run.app
class AppConfig:
    start_anchor = os.environ.get("FRACTURE_START_ANCHOR", "GAGACTGCATGG")
    end_anchor = os.environ.get("FRACTURE_END_ANCHOR", "TTTAGTGAGGGT")
    default_umi = os.environ.get("FRACTURE_DEFAULT_UMI", None)
    assembly_method = os.environ.get("FRACTURE_ASSEMBLY_METHOD", "shortest_path")
    min_coverage = int(os.environ.get("FRACTURE_MIN_COVERAGE", "5"))
    kmer_size = int(os.environ.get("FRACTURE_KMER_SIZE", "10"))
    auto_k = os.environ.get("FRACTURE_AUTO_K", "False").lower() in ('true', '1', 't', 'yes')
    file_path = os.environ.get("FRACTURE_FILE_PATH", None)

config = AppConfig()

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
            create_data_input_sidebar(
                db=db,
                provided_umi_default=config.default_umi or "",
                file_path_default=config.file_path or "jijo.parquet"
            ),
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
                            #                            ui.div(output_widget("unified_graph"), style="min-height: 800px; height: auto; width: 100%")
                            ui.output_ui("graph_display")

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
                        create_assembly_controls(
                            start_anchor_default=config.start_anchor,
                            end_anchor_default=config.end_anchor,
                            assembly_method_default=config.assembly_method,
                            min_coverage_default=config.min_coverage,
                            kmer_size_default=config.kmer_size,
                            auto_k_default=config.auto_k
                        )
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
    current_static_image = reactive.Value(None)
    cached_positions = reactive.Value({})
    layout_cache_key = reactive.Value(None)

    
    # Node selection state management
    clicked_nodes = reactive.Value(set())  # Store clicked node IDs
    current_fig_widget = reactive.Value(None)  # Store current FigureWidget instance
    
    def update_node_selection_colors(fig_widget, selected_nodes, dark_mode):
        """Update node colors directly on existing FigureWidget without full re-render"""
        if fig_widget is None:
            return False
            
        # Find the nodes trace
        nodes_trace_idx = None
        for i, trace in enumerate(fig_widget.data):
            if hasattr(trace, 'name') and trace.name == 'nodes':
                nodes_trace_idx = i
                break
        
        if nodes_trace_idx is None:
            return False
        
        nodes_trace = fig_widget.data[nodes_trace_idx]
        if not hasattr(nodes_trace, 'customdata') or nodes_trace.customdata is None:
            return False
        
        # Create new color arrays
        new_colors = []
        new_line_colors = []
        new_line_widths = []
        
        for i, node_id in enumerate(nodes_trace.customdata):
            node_id_str = str(node_id)
            if selected_nodes and node_id_str in selected_nodes:
                # Selected node styling
                new_colors.append('rgba(255, 0, 0, 0.8)')
                new_line_colors.append('#ff0000')
                new_line_widths.append(4)
            else:
                # Default node styling - preserve original or use default
                if hasattr(nodes_trace.marker, 'color') and i < len(nodes_trace.marker.color):
                    original_color = nodes_trace.marker.color[i]
                    # If it was previously selected (red), revert to default
                    if 'rgba(255, 0, 0' in str(original_color):
                        new_colors.append('rgba(0, 0, 0, 0)' if not dark_mode else 'rgba(255, 255, 255, 0)')
                    else:
                        new_colors.append(original_color)
                else:
                    new_colors.append('rgba(0, 0, 0, 0)' if not dark_mode else 'rgba(255, 255, 255, 0)')
                
                new_line_colors.append('#888' if not dark_mode else '#ccc')
                new_line_widths.append(1)
        
        # Update the trace properties directly
        with fig_widget.batch_update():
            fig_widget.data[nodes_trace_idx].marker.color = new_colors
            fig_widget.data[nodes_trace_idx].marker.line.color = new_line_colors
            fig_widget.data[nodes_trace_idx].marker.line.width = new_line_widths
        
        return True
    
    def get_layout_algorithm(algorithm_name, graph, spring_args, weighted=False):
        """Get the appropriate layout function based on algorithm name"""
        import networkx as nx
        
        if algorithm_name == "fruchterman_reingold":
            return nx.fruchterman_reingold_layout(graph, k=spring_args.get('k'), iterations=spring_args.get('iterations', 50), seed=42)
        elif algorithm_name == "spectral":
            return nx.spectral_layout(graph, scale=spring_args.get('scale', 2.0))
        elif algorithm_name == "random":
            return nx.random_layout(graph, seed=42)
        elif algorithm_name == "circular":
            return nx.circular_layout(graph, scale=spring_args.get('scale', 2.0))
        elif algorithm_name == "shell":
            return nx.shell_layout(graph, scale=spring_args.get('scale', 2.0))
        elif algorithm_name == "spring":
            if weighted:
                return nx.spring_layout(graph, weight='weight', **spring_args, seed=42)
            else:
                return nx.spring_layout(graph, **spring_args, seed=42)
        elif algorithm_name == "kamada_kawai":
            return nx.kamada_kawai_layout(graph.to_undirected(), scale=spring_args.get('scale', 2.0))
        else:
            # Default fallback
            return nx.fruchterman_reingold_layout(graph, k=spring_args.get('k'), iterations=spring_args.get('iterations', 50), seed=42)
    
    # Unified graph state
    current_graph_path = reactive.Value(None)
    graph_source_type = reactive.Value("none")  # "assembly", "upload", or "none"
    ###
    def get_optimized_layout(graph, use_weighted, weight_method, separate_components, 
                            component_padding, min_component_size, spring_args):
        """Calculate optimized layout for a graph, using caching when possible"""
        # Create a cache key based on the graph and layout parameters
        import hashlib
        import networkx as nx
        # Create a hash of the graph structure
        graph_hash = hashlib.md5(str(list(graph.edges())).encode()).hexdigest()
        
        cache_key = (
            graph_hash,
            use_weighted,
            weight_method if use_weighted else None,
            separate_components,
            component_padding if separate_components else None,
            min_component_size if separate_components else None,
            spring_args['k'],
            spring_args['iterations'],
            spring_args['scale']
        )
        
        # Check if we have this layout cached
        if layout_cache_key.get() == cache_key:
            logger.info("Using cached graph layout")
            return cached_positions.get()
        
        logger.info("Calculating new graph layout")
        
        # First try to extract positions from the DOT file
        pos = extract_positions_from_dot(graph)
        
        # If we found positions for most nodes, use them
        if len(pos) >= len(graph.nodes()) * 0.9:  # 90% or more nodes have positions
            logger.info(f"Using positions from DOT file for {len(pos)} nodes")
        else:
            # Otherwise calculate positions based on graph size and parameters
            node_count = len(graph.nodes())
            logger.info(f"Calculating layout for {node_count} nodes")
            
            if input.separate_components():
                # Identify connected components for separate layout
                components = list(nx.connected_components(graph.to_undirected()))
                
                # Filter out components smaller than min_component_size
                filtered_components = [comp for comp in components if len(comp) >= min_component_size]
                
                # Sort components by size (largest first)
                filtered_components.sort(key=len, reverse=True)
                
                # Calculate layout for each component and place in a grid
                pos = {}
                
                # First pass - calculate layouts and store info
                component_info = []
                for component in filtered_components:
                    # Create subgraph for this component
                    subgraph = graph.subgraph(component)
                    
                    # Calculate layout for this component - same as in your visualization code
                    spring_args = {
                        'k': input.layout_k(),
                        'iterations': input.layout_iterations(),
                        'scale': input.layout_scale()
                    }
                    
                    # Use selected algorithm for component layout
                    component_pos = get_layout_algorithm(
                        input.layout_algorithm(), 
                        subgraph, 
                        spring_args, 
                        weighted=input.use_weighted()
                    )
                    
                    # Find bounding box
                    min_x = min(p[0] for p in component_pos.values()) if component_pos else 0
                    max_x = max(p[0] for p in component_pos.values()) if component_pos else 0
                    min_y = min(p[1] for p in component_pos.values()) if component_pos else 0
                    max_y = max(p[1] for p in component_pos.values()) if component_pos else 0
                    width = max_x - min_x
                    height = max_y - min_y
                    
                    # Use the same scale factor calculation
                    import numpy as np
                    scale_factor = np.sqrt(len(component)) / 2.0
                    
                    component_info.append({
                        'component': component,
                        'pos': component_pos,
                        'width': width * scale_factor,
                        'height': height * scale_factor,
                        'scale_factor': scale_factor,
                        'size': len(component)
                    })
                
                # If no components meet the size requirement, revert to standard layout
                if not filtered_components:
                    # Use selected algorithm for full graph when no components meet size requirement
                    pos = get_layout_algorithm(
                        input.layout_algorithm(), 
                        graph, 
                        spring_args, 
                        weighted=input.use_weighted()
                    )
                else:
                    # Second pass - arrange components in a grid, same as in your visualization code
                    current_x, current_y = 0, 0
                    max_height_in_row = 0
                    row_components = 0
                    padding = input.component_padding()
                    max_components_per_row = 3  # Adjust based on your needs
                    
                    for info in component_info:
                        # If we've reached the max components per row, move to next row
                        if row_components >= max_components_per_row:
                            current_x = 0
                            current_y += max_height_in_row + padding
                            max_height_in_row = 0
                            row_components = 0
                        
                        # Position this component
                        for node, node_pos in info['pos'].items():
                            # Scale and shift the position
                            pos[node] = (
                                node_pos[0] * info['scale_factor'] + current_x,
                                node_pos[1] * info['scale_factor'] + current_y
                            )
                        
                        # Update position for next component
                        current_x += info['width'] + padding
                        max_height_in_row = max(max_height_in_row, info['height'])
                        row_components += 1
                # Implement the separate component layout logic
                # (Similar to your existing code)
                # ...
            else:
                # Use selected algorithm regardless of graph size
                logger.info(f"Using {input.layout_algorithm()} layout algorithm for {node_count} nodes")
                pos = get_layout_algorithm(
                    input.layout_algorithm(), 
                    graph, 
                    spring_args, 
                    weighted=use_weighted
                )
        
        # Cache the calculated positions
        cached_positions.set(pos)
        layout_cache_key.set(cache_key)
        
        return pos

    # Helper function to extract positions from DOT file
    def extract_positions_from_dot(graph):
        """Extract position information from the DOT file if available"""
        pos = {}
        for node, attrs in graph.nodes(data=True):
            # Check if position is defined in DOT
            if 'pos' in attrs:
                try:
                    # Parse position string (format is "x,y")
                    pos_str = attrs['pos'].strip('"')
                    x, y = map(float, pos_str.split(','))
                    pos[node] = (x, y)
                except (ValueError, AttributeError):
                    pass
        return pos
    # Theme handling
    @reactive.Effect
    @reactive.event(input.app_theme)
    def update_theme():
        selected_theme = input.app_theme()
        
        if selected_theme == "latte":
            # Update application theme
            #session.send_custom_message("shinyswatch-theme", "minty")
            current_template.set(LATTE_TEMPLATE)
        elif selected_theme == "mocha":
            # Update application theme
            #session.send_custom_message("shinyswatch-theme", "darkly")
            current_template.set(MOCHA_TEMPLATE)
        else:
            # Default slate theme
            #session.send_custom_message("shinyswatch-theme", "slate")
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
    
    def get_node_sequences_from_graph(node_ids):
        """Extract sequences for given node IDs from the current graph"""
        graph_path = current_graph_path.get()
        if not graph_path or not Path(graph_path).exists():
            return {}
        
        try:
            import networkx as nx
            from modules.visualization import parse_node_label
            
            graph = nx.drawing.nx_pydot.read_dot(graph_path)
            node_sequences = {}
            
            for node_id in node_ids:
                if node_id in graph.nodes():
                    attrs = graph.nodes[node_id]
                    label = attrs.get('label', '')
                    if isinstance(label, str):
                        label = label.strip('"').replace('\\\\n', '\n').replace('\\n', '\n')
                        id_part, seq_part, cov_part = parse_node_label(label)
                        if seq_part:
                            node_sequences[node_id] = {
                                'sequence': seq_part.strip(),
                                'coverage': cov_part.strip() if cov_part else 'N/A'
                            }
            
            return node_sequences
        except Exception as e:
            logger.error(f"Error extracting node sequences: {e}")
            return {}

    @output
    @render.ui
    def selected_nodes_table():
        """Display selected nodes with their sequences as a table"""
        try:
            current_nodes = clicked_nodes.get()
            if not current_nodes:
                return ui.div(ui.p("No nodes selected", style="font-style: italic; color: #888;"))
            
            # Get sequences for selected nodes
            node_sequences = get_node_sequences_from_graph(current_nodes)
            
            # Get theme colors
            theme = input.app_theme()
            if theme == "latte":
                bg_color = "#e6e9ef"
                text_color = "#4c4f69"
                secondary_text = "#6c6f85"
                node_bg = "#dce0e8"
                seq_bg = "#ccd0da"
                border_color = "#1e66f5"
                coverage_color = "#40a02b"
            elif theme == "mocha":
                bg_color = "#181825"
                text_color = "#cdd6f4"
                secondary_text = "#a6adc8"
                node_bg = "#313244"
                seq_bg = "#45475a"
                border_color = "#cba6f7"
                coverage_color = "#a6e3a1"
            else:  # dark/slate theme
                bg_color = "#444d56"
                text_color = "#ffffff"
                secondary_text = "#adb5bd"
                node_bg = "#555e67"
                seq_bg = "#6c757d"
                border_color = "#007bff"
                coverage_color = "#28a745"
            
            # Create enhanced table with sequences
            table_rows = []
            for i, node_id in enumerate(sorted(current_nodes), 1):
                # Node ID row
                table_rows.append(
                    ui.div(
                        ui.span(f"{i}. ", style=f"font-weight: bold; color: {border_color};"),
                        ui.span(f"Node: {node_id}", style=f"font-family: monospace; background-color: {node_bg}; color: {text_color}; padding: 2px 6px; border-radius: 3px; font-weight: bold;"),
                        style="margin: 3px 0; display: flex; align-items: center;"
                    )
                )
                
                # Sequence and coverage info
                if node_id in node_sequences:
                    seq_info = node_sequences[node_id]
                    sequence = seq_info['sequence']
                    coverage = seq_info['coverage']
                    
                    # Format sequence as single line
                    formatted_sequence = sequence
                    
                    table_rows.append(
                        ui.div(
                            ui.HTML(f"""
                            <div style="margin-left: 20px; margin-bottom: 8px; color: {text_color};">
                                <div style="margin-bottom: 4px;">
                                    <strong>Coverage:</strong> <span style="color: {coverage_color};">{coverage}</span>
                                </div>
                                <div style="margin-bottom: 4px;">
                                    <strong>Sequence:</strong> <span style="font-size: 11px; color: {secondary_text};">({len(sequence)} bp)</span>
                                </div>
                                <div style="font-family: monospace; background-color: {seq_bg}; color: {text_color}; padding: 8px; border-radius: 4px; word-wrap: break-word; font-size: 12px; line-height: 1.4; user-select: text; cursor: text;">
                                    {formatted_sequence}
                                </div>
                            </div>
                            """)
                        )
                    )
                else:
                    table_rows.append(
                        ui.div(
                            ui.span("No sequence data available", style=f"margin-left: 20px; font-style: italic; color: {secondary_text};"),
                            style="margin: 3px 0;"
                        )
                    )
            
            return ui.div(
                *table_rows,
                style=f"background-color: {bg_color}; color: {text_color}; padding: 10px; border-radius: 5px; border-left: 3px solid {border_color}; max-height: 400px; overflow-y: auto;"
            )
        except Exception as e:
            logger.error(f"Error in selected_nodes_table: {e}")
            return ui.div(ui.p(f"Error: {str(e)}", style="color: red;"))
    
    def get_tab_animation_css():
        """CSS for tab animation effects"""
        return """
        /* Tab shine animation keyframes */
        @keyframes tabShine {
            0% { 
                background: linear-gradient(90deg, transparent, transparent);
                color: inherit;
            }
            25% { 
                background: linear-gradient(90deg, transparent, rgba(255,215,0,0.3), transparent);
                color: #ffd700;
                text-shadow: 0 0 8px rgba(255,215,0,0.6);
            }
            50% { 
                background: linear-gradient(90deg, transparent, rgba(255,215,0,0.5), transparent);
                color: #ffd700;
                text-shadow: 0 0 12px rgba(255,215,0,0.8);
            }
            75% { 
                background: linear-gradient(90deg, transparent, rgba(255,215,0,0.3), transparent);
                color: #ffd700;
                text-shadow: 0 0 8px rgba(255,215,0,0.6);
            }
            100% { 
                background: linear-gradient(90deg, transparent, transparent);
                color: inherit;
            }
        }
        
        @keyframes tabPulse {
            0%, 100% { 
                transform: scale(1); 
                box-shadow: 0 0 0px rgba(255,215,0,0);
            }
            50% { 
                transform: scale(1.05); 
                box-shadow: 0 0 15px rgba(255,215,0,0.4);
            }
        }
        
        /* Classes to apply animations */
        .tab-attention {
            animation: tabShine 2s ease-in-out infinite, tabPulse 2s ease-in-out infinite;
            position: relative;
            transition: all 0.3s ease;
        }
        
        .tab-attention:hover {
            animation: none;
            background: rgba(255,215,0,0.2) !important;
            color: #ffd700 !important;
        }
        
        /* Remove animation after some time */
        .tab-attention-fade {
            animation: tabShine 2s ease-in-out 3, tabPulse 2s ease-in-out 3;
        }
        """

    @output
    @render.ui
    def theme_css():
        theme = input.app_theme()
        base_animations = get_tab_animation_css()
        if theme == "latte":
            return ui.tags.style(base_animations + """
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
            return ui.tags.style(base_animations + """
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
            return ui.tags.style(base_animations + """
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
            logger.info(f"Data columns: {df.columns}")
            
            dataset.set(file_path_or_error)
            data.set(df)
            ui.notification_show(f"Successfully loaded data from {file_path_or_error}")
            logger.info(f"Successfully loaded data from {file_path_or_error}")
            
            # Update UMI choices
            umis = get_umis(df)
            logger.info(f"Found {len(umis)} UMIs")
            ui.notification_show(f"Found {len(umis)} UMIs")

            # Determine which UMI to select by default
            selected_umi = None
            if umis:
                # Use config default if specified and available, otherwise use first UMI
                if config.default_umi and config.default_umi in umis:
                    selected_umi = config.default_umi
                    logger.info(f"Selecting configured default UMI: {selected_umi}")
                else:
                    selected_umi = umis[0]
                    if config.default_umi:
                        logger.warning(f"Configured UMI '{config.default_umi}' not found in data, using first UMI: {selected_umi}")

            ui.update_selectize(
                "umi",
                choices=umis,
                selected=selected_umi
            )

        except Exception as e:
            import traceback
            error_msg = f"Error loading data: {str(e)}"
            logger.error(error_msg)
            logger.info(f"Traceback: {traceback.format_exc()}")
            
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

    def trigger_tab_animation():
        """Trigger the Graph Explorer tab animation to guide user attention"""
        try:
            # Use JavaScript injection instead of custom message
            js_code = """
            $(document).ready(function() {
                // Find the Graph Explorer tab
                var tabLinks = $('a[data-bs-toggle="tab"], a[data-toggle="tab"]');
                var targetTab = null;
                
                tabLinks.each(function() {
                    if ($(this).text().trim() === 'Graph Explorer') {
                        targetTab = $(this);
                        return false;
                    }
                });
                
                if (targetTab) {
                    console.log('Found Graph Explorer tab, applying animation');
                    
                    // Add the animation class
                    targetTab.addClass('tab-attention');
                    
                    // Remove animation after 6 seconds
                    setTimeout(function() {
                        targetTab.removeClass('tab-attention');
                        console.log('Animation removed from tab');
                    }, 6000);
                    
                    // Also remove animation if user clicks the tab
                    targetTab.one('click', function() {
                        $(this).removeClass('tab-attention');
                        console.log('Animation removed due to tab click');
                    });
                } else {
                    console.log('Graph Explorer tab not found');
                }
            });
            """
            
            # Insert the JavaScript code into the page
            ui.insert_ui(
                selector="head",
                where="beforeend",
                ui=ui.tags.script(js_code),
                immediate=True
            )
        except Exception as e:
            logger.error(f"Error triggering tab animation: {e}")
            # Fallback: just show a notification
            ui.notification_show("✨ Check out the Graph Explorer tab!", type="message", duration=3)

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
                
                # Trigger tab animation to guide user to Graph Explorer
                trigger_tab_animation()
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
            error_msg = f"Unexpected error in coverage plot: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.debug(f"Coverage plot traceback: {traceback.format_exc()}")
            
            empty_fig = go.Figure(layout=current_template()['layout'])
            empty_fig.update_layout(
                height=500,
                autosize=True,
                annotations=[dict(
                    text="Coverage plot error - check logs for details",
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
        source_graph_path = f"{base_path}{graph_suffix}__{input.graph_type()}.dot"
        
        if Path(source_graph_path).exists():
            # Copy to central location instead of just setting the path
            central_path = get_central_graph_path()
            try:
                import shutil
                shutil.copy2(source_graph_path, central_path)
                current_graph_path.set(str(central_path))
                graph_source_type.set("assembly")
                ui.notification_show(f"Loaded {input.graph_type()} graph", type="success", duration=2)
            except Exception as e:
                ui.notification_show(f"Error copying graph: {str(e)}", type="error")
        else:
            ui.notification_show(f"Graph file not found: {source_graph_path}", type="error")

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

    # Store uploaded file info for later processing
    uploaded_dot_info = reactive.Value(None)
    
    # Handle DOT file selection (not automatic processing)
    @reactive.Effect
    @reactive.event(input.dot_file)
    def handle_dot_file_selection():
        """Store uploaded DOT file info for processing when Create Graph is clicked"""
        if input.dot_file() is not None:
            file_info = input.dot_file()
            uploaded_dot_info.set(file_info)
            logger.info(f"DOT file selected: {file_info[0]['name']}")
            ui.notification_show("DOT file selected. Click 'Create Graph' to process.", type="message", duration=3)
        else:
            uploaded_dot_info.set(None)
    
    # Handle Create Graph button for uploads
    @reactive.Effect  
    @reactive.event(input.create_graph)
    def handle_create_graph_from_upload():
        """Process uploaded DOT file when Create Graph button is clicked"""
        file_info = uploaded_dot_info.get()
        if file_info is not None:
            uploaded_dot_path = file_info[0]["datapath"]
            logger.info(f"Processing DOT file: {uploaded_dot_path}")
            
            # Copy to central location
            central_path = get_central_graph_path()
            try:
                import shutil
                shutil.copy2(uploaded_dot_path, central_path)
                
                # Set the radio button to upload mode
                ui.update_radio_buttons("graph_source", selected="upload")
                
                current_graph_path.set(str(central_path))
                graph_source_type.set("upload")
                ui.notification_show("Graph created from DOT file", type="success", duration=2)
            except Exception as e:
                ui.notification_show(f"Error processing DOT file: {str(e)}", type="error")
        else:
            ui.notification_show("Please select a DOT file first", type="warning", duration=2)

    # Download handler for static graph PNG
    @render.download(
        filename=lambda: f"fracture_graph_{input.layout_algorithm()}_{int(__import__('time').time())}.png"
    )
    def download_static_graph():
        """Download the current static graph image"""
        image_path = current_static_image.get()
        if image_path and Path(image_path).exists():
            logger.info(f"Downloading static graph: {image_path}")
            # Return the file path directly - Shiny will handle the rest
            return str(image_path)
        else:
            logger.warning("No static graph image available for download")
            # Return empty bytes if no file available
            return b""

    # Update download button state based on static image availability
    @reactive.Effect
    def update_download_button():
        """Enable/disable download button based on static image availability"""
        image_path = current_static_image.get()
        if image_path and Path(image_path).exists():
            # Enable button by removing disabled attribute
            ui.update_action_button("download_static_graph", disabled=False)
        else:
            # Disable button when no image is available
            ui.update_action_button("download_static_graph", disabled=True)

    ####
    @output
    @render.ui
    def graph_display():
        """Display either the static image or the interactive widget based on toggle"""
        if input.use_static_image():
            # If using static image
            if current_static_image.get() is None:
                return ui.div(
                    "Static image not yet generated. Change any graph parameter to generate it.",
                    style="height: 800px; display: flex; align-items: center; justify-content: center; font-style: italic; color: #888;"
                )
            else:
                # Read the image file and convert to a data URL
                try:
                    import base64
                    image_path = current_static_image.get()
                    
                    if not Path(image_path).exists():
                        logger.warning(f"Image file not found: {image_path}")
                        return ui.div(
                            f"Image file not found: {Path(image_path).name}",
                            style="height: 800px; display: flex; align-items: center; justify-content: center; color: red;"
                        )
                    
                    with open(image_path, "rb") as f:
                        image_data = f.read()
                    
                    # Convert to a data URL
                    encoded = base64.b64encode(image_data).decode("utf-8")
                    data_url = f"data:image/png;base64,{encoded}"
                    
                    return ui.div(
                        ui.tags.img(src=data_url, style="width:100%; max-height:800px; object-fit:contain;"),
                        style="display: flex; flex-direction: column; align-items: center; gap: 10px;"
                    )
                except Exception as e:
                    logger.error(f"Error creating data URL: {str(e)}")
                    import traceback
                    logger.debug(f"Data URL error traceback: {traceback.format_exc()}")
                    return ui.div(
                        f"Error loading image: {str(e)}",
                        style="height: 800px; display: flex; align-items: center; justify-content: center; color: red;"
                    )
        else:
            # Using interactive widget
            return ui.div(output_widget("unified_graph"), style=f"height: {input.graph_height()}px;")
        ####

    @reactive.Effect
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
        input.layout_algorithm,
        input.use_static_image,
        input.show_node_labels,
        input.graph_height
    )
    def generate_static_graph():
        """Generate a static PNG image of the graph preserving NetworkX layout"""
        if not input.use_static_image() or current_graph_path.get() is None:
            return
            
        logger.info("Generating static graph image preserving node positions")
        graph_path = current_graph_path.get()

        if graph_path is None:
            logger.warning("No graph path available for static image generation")
            return
            
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            from modules.visualization import create_weighted_graph, extract_coverage
            
            # Set matplotlib to use agg backend (non-interactive)
            mpl.use('agg')
            
            # Read the DOT file
            graph = nx.drawing.nx_pydot.read_dot(graph_path)
            if len(graph.nodes()) == 0:
                logger.warning("Empty graph - no nodes found")
                return
            
            # Apply weighting if needed
            if input.use_weighted():
                graph = create_weighted_graph(graph, input.weight_method())
            
            # Calculate node positions - using the same logic as in your Plotly visualization
            if input.separate_components():
                # Identify connected components for separate layout
                components = list(nx.connected_components(graph.to_undirected()))
                
                # Filter out components smaller than min_component_size
                filtered_components = [comp for comp in components if len(comp) >= input.min_component_size()]
                
                # Sort components by size (largest first)
                filtered_components.sort(key=len, reverse=True)
                
                # Calculate layout for each component and place in a grid
                pos = {}
                
                # First pass - calculate layouts and store info
                component_info = []
                for component in filtered_components:
                    # Create subgraph for this component
                    subgraph = graph.subgraph(component)
                    
                    # Calculate layout for this component - same as in your visualization code
                    spring_args = {
                        'k': input.layout_k(),
                        'iterations': input.layout_iterations(),
                        'scale': input.layout_scale()
                    }
                    
                    # Use selected algorithm for component layout
                    component_pos = get_layout_algorithm(
                        input.layout_algorithm(), 
                        subgraph, 
                        spring_args, 
                        weighted=input.use_weighted()
                    )
                    
                    # Find bounding box
                    min_x = min(p[0] for p in component_pos.values()) if component_pos else 0
                    max_x = max(p[0] for p in component_pos.values()) if component_pos else 0
                    min_y = min(p[1] for p in component_pos.values()) if component_pos else 0
                    max_y = max(p[1] for p in component_pos.values()) if component_pos else 0
                    width = max_x - min_x
                    height = max_y - min_y
                    
                    # Use the same scale factor calculation
                    import numpy as np
                    scale_factor = np.sqrt(len(component)) / 2.0
                    
                    component_info.append({
                        'component': component,
                        'pos': component_pos,
                        'width': width * scale_factor,
                        'height': height * scale_factor,
                        'scale_factor': scale_factor,
                        'size': len(component)
                    })
                
                # If no components meet the size requirement, revert to standard layout
                if not filtered_components:
                    # Use selected algorithm for full graph when no components meet size requirement
                    pos = get_layout_algorithm(
                        input.layout_algorithm(), 
                        graph, 
                        spring_args, 
                        weighted=input.use_weighted()
                    )
                else:
                    # Second pass - arrange components in a grid, same as in your visualization code
                    current_x, current_y = 0, 0
                    max_height_in_row = 0
                    row_components = 0
                    padding = input.component_padding()
                    max_components_per_row = 3  # Adjust based on your needs
                    
                    for info in component_info:
                        # If we've reached the max components per row, move to next row
                        if row_components >= max_components_per_row:
                            current_x = 0
                            current_y += max_height_in_row + padding
                            max_height_in_row = 0
                            row_components = 0
                        
                        # Position this component
                        for node, node_pos in info['pos'].items():
                            # Scale and shift the position
                            pos[node] = (
                                node_pos[0] * info['scale_factor'] + current_x,
                                node_pos[1] * info['scale_factor'] + current_y
                            )
                        
                        # Update position for next component
                        current_x += info['width'] + padding
                        max_height_in_row = max(max_height_in_row, info['height'])
                        row_components += 1
            else:
                # Use the same layout calculation as in your visualization code
                spring_args = {
                    'k': input.layout_k(),
                    'iterations': input.layout_iterations(),
                    'scale': input.layout_scale()
                }
                ###
                # Use selected algorithm for static graph layout (regardless of size)
                logger.info(f"Using {input.layout_algorithm()} layout algorithm for static graph with {len(graph.nodes())} nodes")
                pos = get_layout_algorithm(
                    input.layout_algorithm(), 
                    graph, 
                    spring_args, 
                    weighted=input.use_weighted()
                )


            # Create figure with appropriate styling
            # Calculate height in inches based on pixel height and DPI
            dpi = 150
            height_inches = input.graph_height() / dpi
            # Maintain aspect ratio (12:8 = 1.5:1)
            width_inches = height_inches * 1.5
            plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
            
            # Set theme colors
            dark_mode = (input.app_theme() != "latte")
            bg_color = '#2d3339' if dark_mode else '#ffffff'
            edge_color = '#888888' if dark_mode else '#555555'
            node_color = '#375a7f' if dark_mode else '#3498db'
            text_color = '#ffffff' if dark_mode else '#000000'
            
            # Set background color
            plt.gca().set_facecolor(bg_color)
            plt.gcf().set_facecolor(bg_color)
            
            # Extract node properties for coloring
            colors = []
            sizes = []
            
            # Get selected nodes
            selected_nodes = []
            if input.selected_nodes() and input.selected_nodes().strip():
                selected_nodes.extend([node.strip() for node in input.selected_nodes().split(',') if node.strip()])
            selected_nodes.extend(clicked_nodes.get())
            selected_nodes = list(set(selected_nodes))
            
            # Get coverages for size scaling
            coverages = extract_coverage(graph)
            min_coverage = min(coverages.values()) if coverages else 1
            max_coverage = max(coverages.values()) if coverages else 1
            min_size = 100
            max_size = 600
            
            # Get colors and sizes for each node
            for node in graph.nodes():
                if node in selected_nodes:
                    colors.append('#e74c3c')  # Red for selected
                else:
                    colors.append(node_color)
                
                # Scale node size based on coverage
                if node in coverages:
                    size_factor = (coverages[node] - min_coverage) / max(1, max_coverage - min_coverage)
                    sizes.append(min_size + size_factor * (max_size - min_size))
                else:
                    sizes.append(min_size)
            
            # Draw edges
            nx.draw_networkx_edges(
                graph, pos, 
                alpha=0.7,
                edge_color=edge_color,
                arrows=True,
                arrowsize=10
            )
            
            # Draw nodes
            nx.draw_networkx_nodes(
                graph, pos,
                node_color=colors,
                node_size=sizes,
                alpha=0.9
            )
            
            # Draw labels (conditional based on toggle)
            if input.show_node_labels():
                nx.draw_networkx_labels(
                    graph, pos,
                    font_color=text_color,
                    font_size=8
                )
            
            plt.title("Assembly Graph", color=text_color, fontsize=16)
            plt.axis('off')
            
            # Generate file path for the static image
            base_dir = Path(__file__).parent
            static_dir = base_dir / "www"
            static_dir.mkdir(exist_ok=True)
            
            # Use timestamp to ensure unique filenames
            import time
            timestamp = int(time.time())
            
            # Save with graph source type in filename
            image_filename = f"graph_{graph_source_type.get()}_{timestamp}.png"
            image_path = static_dir / image_filename
            
            # Save the figure as a PNG
            plt.tight_layout()
            plt.savefig(str(image_path), dpi=150, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
            logger.info(f"Saved static graph image to {image_path}")
            
            # Store the current static image path in a reactive value
            current_static_image.set(str(image_path))
            
        except Exception as e:
            logger.error(f"Error generating static graph image: {str(e)}")
            import traceback
            logger.debug(f"Static graph traceback: {traceback.format_exc()}")
        ###
    # Update selection colors without full re-render
    @reactive.effect
    @reactive.event(clicked_nodes, input.selected_nodes, input.selected_sequences)
    def update_selection_colors():
        """Update node selection colors on existing FigureWidget"""
        fig_widget = current_fig_widget.get()
        if fig_widget is None:
            return
        
        # Get all selected nodes (from clicks and text input)
        all_selected_nodes = set(clicked_nodes.get())
        
        if input.selected_nodes() and input.selected_nodes().strip():
            text_nodes = [node.strip() for node in input.selected_nodes().split(',') if node.strip()]
            all_selected_nodes.update(text_nodes)
        
        selected_nodes = list(all_selected_nodes) if all_selected_nodes else None
        dark_mode = (input.app_theme() != "latte")
        
        # Update colors directly
        update_node_selection_colors(fig_widget, selected_nodes, dark_mode)
    
    # Unified graph rendering function (excluding selection dependencies)
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
        input.layout_algorithm,
        input.use_static_image,  # Add this dependency
        input.graph_height
    )
    def unified_graph():
        """Render the unified graph widget"""

        if input.use_static_image():
            # Clear the stored widget when using static
            current_fig_widget.set(None)
            # Return an empty widget
            empty_fig = go.FigureWidget()
            empty_fig.update_layout(
                height=10,  # Minimal height to minimize space
                autosize=True,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                plot_bgcolor='rgba(0,0,0,0)'    # Transparent plot area
            )
            return empty_fig
        
        logger.info(f"Rendering interactive graph - source: {graph_source_type.get()}, path: {current_graph_path.get()}")
        
        # Check if we have a graph to display
        graph_path = current_graph_path.get()
        if graph_path is None:
            current_fig_widget.set(None)
            empty_fig = go.FigureWidget(layout=current_template()['layout'])
            empty_fig.update_layout(
                height=input.graph_height(),
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
            # Parse sequences only (selection handled separately)
            selected_sequences = None
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
                current_fig_widget.set(None)
                error_fig = go.FigureWidget(layout=current_template()['layout'])
                error_fig.update_layout(
                    height=input.graph_height(),
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
            
            # Create the graph plot without selected_nodes (handled separately)
            fig = create_graph_plot(
                graph_path,
                dark_mode=(input.app_theme() != "latte"),
                weighted=input.use_weighted(),
                weight_method=input.weight_method(),
                separate_components=input.separate_components(),
                component_padding=input.component_padding(),
                min_component_size=input.min_component_size(),
                selected_nodes=None,  # Don't include selection in base plot
                selected_sequences=selected_sequences,
                path_nodes=path_nodes,
                layout_algorithm=input.layout_algorithm(),
                spring_args={
                    'k': input.layout_k(),
                    'iterations': input.layout_iterations(),
                    'scale': input.layout_scale()
                },
                start_anchor=input.start_anchor(),
                end_anchor=input.end_anchor()
            )
            
            # Convert to FigureWidget for interactivity
            fig_widget = go.FigureWidget(fig)

            # Set the height based on user input
            fig_widget.update_layout(height=input.graph_height())

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
            
            # Store the FigureWidget for selection updates
            current_fig_widget.set(fig_widget)
            
            return fig_widget
            
        except Exception as e:
            logger.error(f"Error creating graph: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            current_fig_widget.set(None)
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

def get_central_graph_path():
    """Get the central location where all graphs will be stored"""
    base_path = Path(__file__).parent
    central_path = base_path / "data" / "current_assembly_graph.dot"
    # Ensure directory exists
    central_path.parent.mkdir(exist_ok=True, parents=True)
    return central_path








# Add periodic heartbeat to prevent connection timeout
@reactive.Effect
def heartbeat():
    """Send periodic heartbeat to keep connection alive"""
    reactive.invalidate_later(60)  # Trigger every 60 seconds
    # This effect runs silently in background to keep session active

app = App(app_ui, server)
