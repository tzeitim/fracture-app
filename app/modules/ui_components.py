# UI component definitions
from shiny import ui
from .config import SYSTEM_PREFIXES

def create_data_input_sidebar(db=None, system_prefixes=None):
    """Create the data input sidebar panel"""
    if system_prefixes is None:
        system_prefixes = SYSTEM_PREFIXES
    
    if db is None:
        db = {}
        
    return ui.panel_well(
        ui.h1("Hall of Fame"),
        ui.HTML("GGGCCTGCATCGCACTGTGG google<br>"),
        ui.HTML("TTTTCCCGACGGCTGATCGG messy<br>"),
        ui.HTML("AACATGGACGGTACATCGGG great example of horror<br>"),
        ui.HTML("AACCCCAGAGGCTCAAGTGG full<br>"),
        ui.HTML("GCTCGTATCCCGAAGCTAGG failed but overlaps<br>"),
        ui.HTML("GATGCCTACCATCACTGTGG failed but overlaps<br>"),
        ui.HTML("ATTGGCGGCACACTGTCCTG tricky <br>"),
        #---
        ui.input_numeric("insert_size", "Lineage reporter reference length (bp)", value=450),
        ui.input_numeric("sample_n_umis", "Number of UMIs to sample", value=100),
        ui.input_checkbox(
            "sample_umis",
            "sub-sample UMIs?",
            value=True
            ),
        ui.input_numeric("sample_min_reads", "Minimum number of reads per umi", value=100),
        ui.input_text("provided_umi", "Provided UMI (optional)", value="", placeholder="Enter specific UMI to load"),
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

def create_graph_controls():
    """Create the graph control panel"""
    return ui.panel_well(
        ui.h4("Assembly Graph"),
        ui.input_action_button(
            "draw_graph",
            "Draw Graph",
            class_="btn-primary"
            ),
        ui.panel_well(
            ui.input_select(
                "graph_type",
                "Graph Plot Settings",
                choices={
                    "compressed": "Compressed Graph",
                    "preliminary": "Preliminary Graph"
                    },
                selected="compressed"
                ),
            ),
        ui.panel_well(
            ui.h4("Graph Layout Options"),
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
                )
            ),
        ui.panel_well(
            ui.h4("Graph Layout Parameters"),
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
                value=3,
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
