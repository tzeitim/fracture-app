from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_plotly
import shinyswatch
import polars as pl
import os
from pathlib import Path
import plotly.graph_objects as go
import matplotlib
import pandas as pd

# Initialize matplotlib
matplotlib.use("agg")

# Import modules
from modules.config import SYSTEM_PREFIXES, DARK_TEMPLATE
from modules.data_loader import load_database, load_data
from modules.data_processing import get_umis, get_selected_umi_stats, compute_coverage, assemble_umi, sweep_assembly_params
from modules.visualization import (
    format_top_contigs_table, create_graph_plot, create_coverage_distribution_plot, 
    create_reads_per_umi_plot
)
from modules.ui_components import (
    create_data_input_sidebar, create_assembly_controls, 
    create_graph_controls, create_parameter_sweep_controls
)

# Configure polars for pretty printing
pl.Config().set_fmt_str_lengths(666)

# Load database from file
try:
    db = load_database('../parquet_db.txt')
except Exception as e:
    print(f"Error loading database: {e}")
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
            create_data_input_sidebar(db),
            ui.panel_well(
                ui.hr(),
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
            ui.nav_panel("Assembly Results",
                ui.row(
                    ui.panel_well(
                        ui.h4("Selection UMI Stats"),
                        ui.output_ui("selected_umi_stats")
                    )
                ),
                ui.row(
                    ui.column(6,
                        ui.panel_well(
                            ui.output_ui("assembly_stats"),
                        ),
                    ),
                    ui.column(6,
                        ui.h4("Coverage Plot (from uniqued reads!!)"),
                        ui.div(output_widget("coverage_plot"), style="height: 500px;"),
                    ),
                ),
                ui.row(
                    ui.column(2, 
                        create_assembly_controls(),
                        create_graph_controls()
                    ),
                    ui.column(10,
                        ui.panel_well(
                            ui.div(output_widget("assembly_graph"), style="min-height: 1000px; height: auto; width: 100%")
                        )
                    ),
                    ui.column(2,)
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
            selected="Assembly Results"  
        )
    ),
    theme=shinyswatch.theme.slate()
)

def server(input, output, session):
    # Clean up any temporary files
    os.system("rm -f *__*.dot *__*.csv")

    # Reactive data storage
    data = reactive.Value(None)
    assembly_result = reactive.Value(None)
    sweep_results = reactive.Value(None)
    path_results = reactive.Value(None)
    dataset = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.parquet_file_local, input.parquet_file, input.remote_path, input.input_type, input.system_prefix, input.sample_n_umis, input.sample_min_reads, input.sample_umis)
    def load_dataset():
        with reactive.isolate():
            data.set(None)
            sweep_results.set(None) 
            assembly_result.set(None)
        ui.update_selectize("umi", choices=[], selected=None)

        try:
            # Determine input type and get file_info
            input_type = input.input_type()
            file_info = None
            if input_type == "upload":
                file_info = input.parquet_file()
            elif input_type == "local":
                file_info = input.parquet_file_local()
            
            # Load data based on input type
            df, file_path_or_error = load_data(
                input_type, 
                file_info,
                system_prefix=input.system_prefix() if input_type == "remote" else None,
                remote_path=input.remote_path() if input_type == "remote" else None,
                sample_umis=input.sample_umis(),
                sample_n_umis=input.sample_n_umis(),
                sample_min_reads=input.sample_min_reads()
            )
            
            if df is None:
                ui.notification_show(file_path_or_error, type='error')
                return
                
            # Successfully loaded data
            dataset.set(file_path_or_error)
            data.set(df)
            ui.notification_show(f"Successfully loaded data from {file_path_or_error}")
            
            # Update UMI choices
            umis = get_umis(df)
            ui.update_selectize(
                "umi",
                choices=umis,
                selected=umis[0] if umis else None
            )

        except Exception as e:
            with reactive.isolate():
                data.set(None)
                sweep_results.set(None)
                assembly_result.set(None)
            ui.notification_show(f"Error loading data: {str(e)}", type="error")

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
            return ui.HTML(f"""
            Dataset: <span style="color: #ffffff;">{dataset()}</span>
            Reads per UMI: <span style="color: #ffffff;">{umi_reads}</span>
            """
                       )
        except Exception as e:
            print(f"Error in selected_umi_stats: {e}")  # Log error
            return "Error calculating UMI stats"
            
    @output
    @render_plotly
    def coverage_dist():
        return create_coverage_distribution_plot(data())

    @output
    @render_plotly
    def reads_per_umi():
        return create_reads_per_umi_plot(data())

    @reactive.Effect
    @reactive.event(input.assemble)
    def run_regular_assembly():
        try:
            if data() is None or not input.umi():
                ui.notification_show("Please load data and select a UMI first", type="warning")
                return

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
            
            handle_assembly_result(result)
        except Exception as e:
            ui.notification_show(f"Error in regular assembly: {str(e)}", type="error")

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
                        print(f'Loading path file {path_file} ... ')
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
        return ui.HTML(f"""
    Target UMI: <h4>{input.umi()}</h4>
    Assembly Method: {input.assembly_method()}
    Contig Length: <span style="color: #ff8080;">{len(assembly_result())}</span> 
    Input Reads: <span style="color: #ffffff;">{read_count}</span> 
    K-mer Size: <span style="color: #ffa07a;">{input.kmer_size()}</span> 
    Min Coverage: <span style="color: #ffa07a;">{input.min_coverage()}</span>"""
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
            print(f"Error loading top contigs: {str(e)}")
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

            print(f"Starting sweep with parameters:")
            print(f"UMI: {input.umi()}")
            print(f"K-mer range: {k_start}-{k_end}, step={input.k_step()}")
            print(f"Coverage range: {cov_start}-{cov_end}")

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
            print(f"Sweep error details: {str(e)}")
            ui.notification_show(
                f"Error during parameter sweep: {str(e)}", 
                type="error"
            )

            # Reset sweep results
            sweep_results.set(None)

    @output
    @render_plotly
    @reactive.event(input.umi, data, assembly_result)
    def coverage_plot():
        if data() is None or not input.umi() or input.umi() not in data()['umi']:
            empty_fig = go.Figure(layout=DARK_TEMPLATE['layout'])
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
            # Load reference sequence for alignment
            mods = pl.read_parquet(f"{Path(__file__).parent}/mods.parquet")
            ref_str = mods.filter(pl.col('mod')=='mod_0')['seq'][0]
            
            # Compute coverage
            result = compute_coverage(data(), input.umi(), ref_str)
            
            # Create plot
            fig = px.line(
                    result,
                    x='covered_positions',
                    y='coverage',
                    labels=dict(
                        x="Position in reference",
                        y="Reads",
                        ),
                    title="Coverage of raw reads",
                    height=500,
                    )

            fig.update_layout(
                    **DARK_TEMPLATE['layout'],
                    xaxis_title="Position",
                    yaxis_title="Coverage",
                    autosize=True,
                    )

            # Ensure axis labels are visible
            fig.update_xaxes(showticklabels=True, title_standoff=25)
            fig.update_yaxes(showticklabels=True, title_standoff=25)

            return fig
        except Exception as e:
            print(f"Error creating coverage plot: {e}")
            empty_fig = go.Figure(layout=DARK_TEMPLATE['layout'])
            empty_fig.update_layout(
                height=500,
                autosize=True,
                annotations=[dict(
                    text=f"Error creating coverage plot: {str(e)}",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color='#ffffff', size=14)
                )]
            )
            return empty_fig

    @output
    @render_plotly
    def sweep_heatmap():
        if sweep_results() is None:
            # Create empty figure with specific dimensions
            empty_fig = go.Figure(layout=DARK_TEMPLATE['layout'])
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
                **DARK_TEMPLATE['layout'],
                xaxis_title="Minimum Coverage",
                yaxis_title="K-mer Size",
                coloraxis_colorbar_title="Contig Length"
                )

        # Ensure axis labels are visible
        fig.update_xaxes(showticklabels=True, title_standoff=25)
        fig.update_yaxes(showticklabels=True, title_standoff=25)

        return fig

    @output
    @render_plotly
    @reactive.event(input.use_weighted, input.draw_graph, input.assemble)
    def assembly_graph():
        if assembly_result() is None:
            empty_fig = go.Figure(layout=DARK_TEMPLATE['layout'])
            empty_fig.update_layout(
                    height=1000,
                    autosize=True,
                    annotations=[dict(
                        text="No assembly graph available",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(color='#ffffff', size=14)
                        )]
                    )
            return empty_fig

        # Build path based on selected graph type and assembly mode
        base_path = f"{Path(__file__).parent}/"
        graph_suffix = input.umi()
        graph_path = f"{base_path}{graph_suffix}__{input.graph_type()}.dot"

        print(f"Looking for graph at: {graph_path}")  

        if not Path(graph_path).exists():
            print(f"Graph file not found: {graph_path}")
            empty_fig = go.Figure(layout=DARK_TEMPLATE['layout'])
            empty_fig.update_layout(
                    height=1000,
                    autosize=True,
                    annotations=[dict(
                        text=f"Graph file not found: {graph_path}",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(color='#ffffff', size=14)
                        )]
                    )
            return empty_fig        

        path_nodes = None
        if path_results() is not None and isinstance(path_results(), dict):
            path_nodes = path_results().get('path_nodes')

        # Create graph visualization
        fig = create_graph_plot(
                graph_path, 
                dark_mode=True,
                line_shape='linear',
                graph_type=input.graph_type(),
                path_nodes=path_nodes,
                weighted=input.use_weighted(),
                weight_method=input.weight_method(),
                spring_args={
                    'k':input.layout_k(),
                    'iterations':input.layout_iterations(),
                    'scale':input.layout_scale(),
                    },
                )

        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                    autosize=True,
                    **DARK_TEMPLATE['layout'],
                    annotations=[dict(
                        text="Error loading assembly graph",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(color='#ffffff', size=14)
                        )]
                    )

        # Add debug information to layout
        fig.update_layout(
                annotations=[
                    dict(
                        text=f"Graph: {graph_path}<br>",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0,  # Position below the plot (-0.1 for some spacing)
                        showarrow=False,
                        font=dict(size=10, color='#ffffff'),
                        xanchor='center',
                        yanchor='top'
                        )
                    ],
                margin=dict(b=60),  # Increase bottom margin to accommodate the text
                autosize=True,
                height=1000,
                )

        return fig

app = App(app_ui, server)