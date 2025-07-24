# UI component definitions
from shiny import ui
from .config import SYSTEM_PREFIXES


def create_data_input_sidebar(db=None, system_prefixes=None):
    """Create the data input sidebar panel"""
    if system_prefixes is None:
        system_prefixes = SYSTEM_PREFIXES
    
    if db is None:
        db = {}
        
    return ui.navset_tab(
        ui.nav_panel("Data",
            ui.input_numeric("insert_size", "Lineage reporter reference length (bp)", value=450),
            ui.input_numeric("sample_n_umis", "Number of UMIs to sample", value=100),
            ui.input_checkbox(
                "sample_umis",
                "sub-sample UMIs?",
                value=True
                ),
            ui.input_numeric("sample_min_reads", "Minimum number of reads per umi", value=100),
            ui.input_text("provided_umi", "Provided UMI (optional)", value="", placeholder="Enter specific UMI to load"),
            
            # Coverage Plot Controls
            ui.input_checkbox(
                "enable_coverage_plot",
                "Enable Coverage Plot",
                value=True
                ),
            ui.input_text(
                "reference_sequence",
                "Reference Sequence (optional)",
                value="",
                placeholder="Leave empty to use default from mods.parquet"
                ),
            #---
            ui.hr(),
            ui.h3("Data Input"),
            ui.input_radio_buttons(
                "input_type",
                "Select Input Method",
                {
                    "upload": "Upload File",
                    "remote": "Remote File Path",
                    "local": "Local File Path"
                    },
                selected="local"
                ),
            ui.panel_conditional(
                "input.input_type === 'remote'",

                ui.input_selectize(
                    "system_prefix",
                    "Select System",
                    choices=system_prefixes,
                    selected="laptop"
                    ),
                ui.input_selectize(
                    "remote_path", 
                    "Select Dataset",
                    choices=db,
                    ),
                ),

            ui.panel_conditional(
                "input.input_type === 'upload'",
                ui.input_file("parquet_file", "Upload Parquet File", accept=[".parquet"])
                ),

            ui.panel_conditional(
                "input.input_type === 'local'",
                ui.input_text("parquet_file_local", "Local Parquet File", value='jijo.parquet')
                ),
        ),
        ui.nav_panel("Hall of Fame",
            ui.h4("Famous UMIs"),
            ui.HTML("GGGCCTGCATCGCACTGTGG google<br>"),
            ui.HTML("TTTTCCCGACGGCTGATCGG messy<br>"),
            ui.HTML("AACATGGACGGTACATCGGG great example of horror<br>"),
            ui.HTML("AACCCCAGAGGCTCAAGTGG full<br>"),
            ui.HTML("GCTCGTATCCCGAAGCTAGG failed but overlaps<br>"),
            ui.HTML("GATGCCTACCATCACTGTGG failed but overlaps<br>"),
            ui.HTML("ATTGGCGGCACACTGTCCTG tricky <br>"),
        ),
        ui.nav_panel("Graph",
            create_graph_source_controls(),
            create_node_selection_controls(),
            create_graph_safety_controls(),  
            create_graph_controls_with_safety(),
        ),
    )

def create_assembly_controls():
    """Create the assembly control panel"""
    return ui.panel_well(
        ui.input_radio_buttons(
            "assembly_method",
            "Select Assembly Method",
            {
                "compression": "Graph Compression",
                "shortest_path": "Shortest Path"
                },
            selected="shortest_path"
            ),
        ui.input_action_button(
            "assemble",
            "Assemble Contig",
            class_="btn-primary"
            ),

        ui.input_text("start_anchor", "Start Anchor", value="GAGACTGCATGG", placeholder="Sequence at the 5' end"),
        ui.input_text("end_anchor", "End Anchor", value="TTTAGTGAGGGT", placeholder="Sequence at the 3' end"),
        ui.input_selectize(
            "umi", 
            "Select UMI",
            choices=[]
            ),
        ui.input_numeric(
            "min_coverage",
            "Minimum Coverage",
            value=5,
            ),
        ui.input_numeric(
            "kmer_size", 
            "K-mer Size", 
            value=10,
            ),
        ui.input_checkbox(
            "auto_k",
            "Auto K-mer Size",
            value=False
            ),
    )

def create_graph_source_controls():
    """Create controls for selecting graph source"""
    return ui.panel_well(
        ui.h4("Graph Source"),
        ui.input_radio_buttons(
            "graph_source",
            "Select graph source:",
            choices={
                "assembly": "Generate from Assembly",
                "upload": "Upload DOT File"
            },
            selected="assembly"
        ),
        
        # Show assembly controls when "assembly" is selected
        ui.panel_conditional(
            "input.graph_source === 'assembly'",
            ui.hr(),
            ui.h5("Assembly Settings"),
            ui.input_select(
                "graph_type",
                "Graph Type:",
                choices={"compressed": "Compressed", "preliminary": "Preliminary"},
                selected="compressed"
            ),
            ui.input_action_button(
                "generate_graph",
                "Generate Graph",
                class_="btn-primary"
            )
        ),
        
        # Show upload controls when "upload" is selected
        ui.panel_conditional(
            "input.graph_source === 'upload'",
            ui.hr(),
            ui.h5("Upload DOT File"),
            ui.input_file(
                "dot_file",
                "Choose .dot file:",
                accept=[".dot"],
                multiple=False
            )
        ),
    )


def create_parameter_sweep_controls():
    """Create the parameter sweep control panel"""
    return ui.panel_well(
        ui.h3("Parameter Sweep"),
        ui.input_slider(
            "k_start",
            "K-mer Range Start",
            min=5,
            max=14,
            value=5,
            step=1
            ),

        ui.input_slider(
            "k_end",
            "K-mer Range End",
            min=15,
            max=64,
            value=15,
            step=1
            ),

        ui.input_slider(
            "cov_start",
            "Coverage Range Start",
            min=1,
            max=17,
            value=1
            ),

        ui.input_slider(
            "cov_end",
            "Coverage Range End",
            min=18,
            max=500,
            value=20,
            ),

        ui.input_slider(
            "k_step",
            "K-mer step",
            min=1,
            max=20,
            value=1,
            step=1
            ),

        #--
        ui.input_action_button(
            "run_sweep",
            "Run Parameter Sweep",
            class_="btn-primary"
            ),
        ui.hr(),
    )

def create_node_selection_controls():
    """Create unified node selection controls"""
    return ui.panel_well(
        ui.h4("Node Selection"),
        ui.p("Click nodes in the graph or enter IDs/sequences below"),
        
        ui.input_text(
            "selected_nodes",
            "Node IDs (comma-separated):",
            value="",
            placeholder="e.g., n0, n1, n597"
        ),
        
        ui.input_text(
            "selected_sequences",
            "Sequences (comma-separated):",
            value="",
            placeholder="e.g., ATGCGTACGT, GCTAGCATCG"
        ),
        
        ui.input_action_button(
            "clear_selection",
            "Clear Selection",
            class_="btn-secondary"
        ),
    )

def create_theme_controls():
    """Create theme selection controls"""
    return ui.div(
        ui.h4("App Theme"),
        ui.p("Select a color theme for the application interface:"),
        ui.input_radio_buttons(
            "app_theme",
            "Select Theme",
            {
                "slate": "Slate (Default)",
                "latte": "Catppuccin Latte (Light)",
                "mocha": "Catppuccin Mocha (Dark)"
            },
            selected="slate",
            inline=False,  # Stack vertically for better visibility
            width="100%"   # Ensure full width
        ),
        ui.div(
            ui.p("Changes are applied immediately across the entire application."),
            style="margin-top: 20px; font-style: italic;"
        )
    )


# Addition to modules/ui_components.py

def create_graph_safety_controls():
    """Create controls for graph rendering safety mechanism"""
    return ui.panel_well(
        ui.h4("Large Graph Safety Controls"),
        
        # Node threshold slider
        ui.input_slider(
            "safety_node_threshold",
            "Node Count Threshold:",
            min=100,
            max=10000,
            value=1000,
            step=100,
            post=" nodes",
            ticks=True
        ),
        
        # Rendering mode selector
        ui.input_radio_buttons(
            "graph_render_mode",
            "Rendering Mode:",
            choices={
                "auto": "Auto (Recommended)",
                "full": "Full Interactive",
                "simplified": "Simplified Interactive", 
                "static": "Static (Fast)"
            },
            selected="auto"
        ),
        
        # Show mode description
        ui.panel_conditional(
            "input.graph_render_mode === 'auto'",
            ui.div(
                ui.tags.i(className="bi bi-info-circle"),
                " Automatically selects rendering mode based on graph size",
                style="color: #6c757d; font-size: 0.9em; margin-top: 5px;"
            )
        ),
        ui.panel_conditional(
            "input.graph_render_mode === 'full'",
            ui.div(
                ui.tags.i(className="bi bi-exclamation-triangle"),
                " Full interactivity - may be slow for large graphs",
                style="color: #ffc107; font-size: 0.9em; margin-top: 5px;"
            )
        ),
        ui.panel_conditional(
            "input.graph_render_mode === 'simplified'",
            ui.div(
                ui.tags.i(className="bi bi-speedometer2"),
                " Reduced features for better performance",
                style="color: #17a2b8; font-size: 0.9em; margin-top: 5px;"
            )
        ),
        ui.panel_conditional(
            "input.graph_render_mode === 'static'",
            ui.div(
                ui.tags.i(className="bi bi-lightning"),
                " Fastest rendering, no interactivity",
                style="color: #28a745; font-size: 0.9em; margin-top: 5px;"
            )
        ),
        
        ui.hr(),
        
        # Advanced options (collapsible)
        ui.input_switch(
            "show_advanced_safety",
            "Show Advanced Options",
            value=False
        ),
        
        ui.panel_conditional(
            "input.show_advanced_safety === true",
            ui.div(
                ui.input_numeric(
                    "static_sample_size",
                    "Static Mode Sample Size:",
                    value=1000,
                    min=100,
                    max=5000,
                    step=100
                ),
                ui.p(
                    "Number of nodes to sample when rendering very large graphs in static mode",
                    style="font-size: 0.85em; color: #6c757d;"
                ),
                
                ui.input_checkbox(
                    "force_fast_layout",
                    "Force fast layout algorithm",
                    value=False
                ),
                ui.p(
                    "Use circular layout instead of spring layout for graphs over 500 nodes",
                    style="font-size: 0.85em; color: #6c757d;"
                )
            )
        ),
        
        # Graph size indicator
        ui.output_ui("graph_size_indicator")
    )


def create_graph_controls_with_safety():
    """Create the enhanced graph display control panel with safety options"""
    return ui.panel_well(
        ui.h4("Graph Display Options"),
        
        # Add safety status at the top
        ui.output_ui("safety_status"),
        ui.hr(),
        
        ui.input_switch(
            "use_weighted",
            "Use Weighted Layout",
            value=False
        ),
        ui.input_select(
            "weight_method",
            "Weight Calculation Method",
            choices={
                "nlog": "Negative Log Coverage",
                "coverage": "Direct Coverage"
            },
            selected="nlog"
        ),
        
        # Component controls with safety warning
        ui.div(
            ui.input_switch(
                "separate_components",
                "Position Disjoint Graphs Separately",
                value=True
            ),
            ui.panel_conditional(
                "output.is_large_graph === true",
                ui.div(
                    ui.tags.i(className="bi bi-exclamation-circle"),
                    " Disabled for large graphs",
                    style="color: #dc3545; font-size: 0.85em;"
                )
            )
        ),
        
        ui.input_numeric(
            "component_padding",
            "Component Padding",
            value=3.0,
            min=0.5,
            max=10.0,
            step=0.5
        ),
        ui.input_numeric(
            "min_component_size",
            "Minimum Component Size to Display",
            value=3,
            min=1,
            max=50
        ),
        
        ui.hr(),
        ui.h5("Spring Layout Parameters"),
        ui.input_numeric(
            "layout_k",
            "Spring Constant (k)",
            value=1.5,
            min=0.1,
            max=5.0,
            step=0.1
        ),
        ui.input_numeric(
            "layout_iterations", 
            "Layout Iterations",
            value=50,
            min=10,
            max=200,
            step=10
        ),
        ui.input_numeric(
            "layout_scale",
            "Layout Scale",
            value=2.0,
            min=0.5,
            max=10.0,
            step=0.5
        )
    )


# UI helper components for the main app

def graph_size_indicator_ui(size_info):
    """Create a visual indicator for graph size"""
    node_count = size_info.get('node_count', 0)
    edge_count = size_info.get('edge_count', 0)
    
    # Determine color based on size
    if node_count > 5000:
        color = "#dc3545"  # danger red
        icon = "exclamation-triangle-fill"
        status = "Very Large"
    elif node_count > 1000:
        color = "#ffc107"  # warning yellow
        icon = "exclamation-circle-fill"
        status = "Large"
    else:
        color = "#28a745"  # success green
        icon = "check-circle-fill"
        status = "Normal"
    
    return ui.div(
        ui.h5("Graph Size", style="margin-bottom: 10px;"),
        ui.div(
            ui.tags.i(className=f"bi bi-{icon}", style=f"color: {color}; margin-right: 5px;"),
            f"{status} Graph",
            style=f"color: {color}; font-weight: bold;"
        ),
        ui.p(f"Nodes: {node_count:,}", style="margin: 5px 0;"),
        ui.p(f"Edges: {edge_count:,}", style="margin: 5px 0;"),
        style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;"
    )


def safety_status_ui(size_info, render_mode):
    """Create a status indicator for current safety mode"""
    if size_info.get('requires_safety', False):
        return ui.div(
            ui.tags.i(className="bi bi-shield-check", style="color: #17a2b8; margin-right: 5px;"),
            f"Safety mode active: {render_mode} rendering",
            style="background-color: #d1ecf1; border: 1px solid #bee5eb; "
                  "color: #0c5460; padding: 5px 10px; border-radius: 3px; "
                  "font-size: 0.9em; margin-bottom: 10px;"
        )
    else:
        return ui.div()  # Empty div when safety not needed
