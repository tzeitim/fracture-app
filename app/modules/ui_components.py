# UI component definitions
from shiny import ui
from .config import SYSTEM_PREFIXES

def create_data_input_sidebar(db=None, system_prefixes=None, provided_umi_default="", file_path_default="jijo.parquet"):
    """Create the data input sidebar panel

    Args:
        db: Database dictionary (optional)
        system_prefixes: System prefix dictionary (optional)
        provided_umi_default: Default value for provided UMI input (optional)
        file_path_default: Default value for local file path (optional)
    """
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
            ui.div(
                ui.input_text("provided_umi", "Provided UMI (optional)", value=provided_umi_default, placeholder="Enter specific UMI to load"),
                ui.input_action_button(
                    "clear_provided_umi",
                    "Clear",
                    class_="btn-sm btn-outline-secondary",
                    style="margin-top: 5px; margin-bottom: 10px;"
                ),
            ),

            # Coverage Plot Controls
            ui.input_checkbox(
                "enable_coverage_plot",
                "Enable Coverage Plot",
                value=True
                ),
            ui.div(
                ui.input_text(
                    "reference_sequence",
                    "Reference Sequence (optional)",
                    value="",
                    placeholder="Leave empty to use default from mods.parquet"
                    ),
                ui.input_action_button(
                    "clear_reference_sequence",
                    "Clear",
                    class_="btn-sm btn-outline-secondary",
                    style="margin-top: 5px; margin-bottom: 10px;"
                ),
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
                ui.div(
                    ui.input_text("parquet_file_local", "Local Parquet File", value=file_path_default),
                    ui.input_action_button(
                        "clear_parquet_file_local",
                        "Clear",
                        class_="btn-sm btn-outline-secondary",
                        style="margin-top: 5px;"
                    ),
                )
                ),
        ),
        # Hall of Fame tab - disabled but preserved for future use
        # ui.nav_panel("Hall of Fame",
        #     ui.h4("Famous UMIs"),
        #     ui.HTML("GGGCCTGCATCGCACTGTGG google<br>"),
        #     ui.HTML("TTTTCCCGACGGCTGATCGG messy<br>"),
        #     ui.HTML("AACATGGACGGTACATCGGG great example of horror<br>"),
        #     ui.HTML("AACCCCAGAGGCTCAAGTGG full<br>"),
        #     ui.HTML("GCTCGTATCCCGAAGCTAGG failed but overlaps<br>"),
        #     ui.HTML("GATGCCTACCATCACTGTGG failed but overlaps<br>"),
        #     ui.HTML("ATTGGCGGCACACTGTCCTG tricky <br>"),
        # ),
        ui.nav_panel("Graph",
            create_graph_source_controls(),
            create_node_selection_controls(),
            create_graph_controls()
        ),
    )

def create_assembly_controls(start_anchor_default="GAGACTGCATGG", end_anchor_default="TTTAGTGAGGGT",
                            assembly_method_default="shortest_path", min_coverage_default=5,
                            kmer_size_default=10, auto_k_default=False):
    """Create the assembly control panel

    Args:
        start_anchor_default (str): Default value for Start Anchor input
        end_anchor_default (str): Default value for End Anchor input
        assembly_method_default (str): Default assembly method ('compression' or 'shortest_path')
        min_coverage_default (int): Default minimum coverage value
        kmer_size_default (int): Default k-mer size
        auto_k_default (bool): Default auto k-mer setting
    """
    return ui.panel_well(
        ui.input_radio_buttons(
            "assembly_method",
            "Select Assembly Method",
            {
                "compression": "Graph Compression",
                "shortest_path": "Shortest Path"
                },
            selected=assembly_method_default
            ),
        ui.input_action_button(
            "assemble",
            "Assemble Contig",
            class_="btn-primary"
            ),

        ui.input_text("start_anchor", "Start Anchor", value=start_anchor_default, placeholder="Sequence at the 5' end"),
        ui.input_text("end_anchor", "End Anchor", value=end_anchor_default, placeholder="Sequence at the 3' end"),
        ui.input_selectize(
            "umi",
            "Select UMI",
            choices=[]
            ),
        ui.input_numeric(
            "min_coverage",
            "Minimum Coverage",
            value=min_coverage_default,
            ),
        ui.input_numeric(
            "kmer_size",
            "K-mer Size",
            value=kmer_size_default,
            ),
        ui.input_checkbox(
            "auto_k",
            "Auto K-mer Size",
            value=auto_k_default
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
        ui.input_select(
            "layout_algorithm",
            "Layout Algorithm:",
            choices={
                "kamada_kawai": "Kamada-Kawai (Best quality - Recommended)",
                "fruchterman_reingold": "Fruchterman-Reingold (Balanced)",
                "spectral": "Spectral (Fast for large graphs)",
                "spring": "Spring (High quality, slow)",
                "circular": "Circular (Very fast)",
                "shell": "Shell (Fast, concentric)",
                "random": "Random (Fastest)"
            },
            selected="kamada_kawai"
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
            ),
            ui.input_action_button(
                "create_graph",
                "Create Graph",
                class_="btn-primary"
            )
        ),
    )

def create_graph_controls():
    """Create the graph display control panel"""
    return ui.panel_well(
        ui.h4("Graph Display Options"),
        ui.input_switch(
            "use_static_image",
            "Use Static Image (PNG)",
            value=True  
            ),
        ui.input_switch(
            "show_node_labels",
            "Show Node Labels in PNG",
            value=False
            ),
        ui.download_button(
            "download_static_graph", 
            "Download PNG",
            icon="download"
        ),
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
                "inverse": "Inverse Coverage"
                },
            selected="inverse"
            ),
        ui.hr(),
        ui.h5("Layout Parameters"),
        ui.input_switch(
            "separate_components",
            "Separate Disjoint Graphs",
            value=False
            ),
        ui.input_numeric(
            "component_padding",
            "Component Spacing",
            value=3.0,
            min=0.5,
            max=10.0,
            step=0.5
            ),
        ui.input_numeric(
            "min_component_size",
            "Min Component Size",
            value=1,
            min=1,
            max=20,
            step=1
            ),
        ui.input_numeric(
            "layout_k",
            "Node Spacing (k)",
            value=0.001,
            min=0,
            max=5.0,
            step=0.001,
            ),
        ui.input_numeric(
            "layout_iterations",
            "Layout Iterations",
            value=500,
            min=10,
            max=2000,
            step=10
            ),
        ui.input_numeric(
            "layout_scale",
            "Layout Scale",
            value=2.0,
            min=0.5,
            max=5.0,
            step=0.5
            ),
        ui.input_numeric(
            "graph_height",
            "Graph Height (px)",
            value=1200,
            min=400,
            max=3000,
            step=100
            ),
        ui.input_numeric(
            "graph_width",
            "Graph Width (px)",
            value=1600,
            min=600,
            max=4000,
            step=100
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
