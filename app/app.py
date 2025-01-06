from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_plotly
import shinyswatch
import polars as pl
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from ogtk.ltr.fracture.pipeline import api_ext
def create_graph_plot(dot_path, dark_mode=True, line_shape='linear', graph_type='compressed'):
    """Convert a DOT file to a Plotly figure with optimized layout settings.
    
    Args:
        dot_path (str): Path to the DOT file
        dark_mode (bool): Whether to use dark theme colors
        
    Returns:
        go.Figure: Plotly figure object
    """
    import networkx as nx
    from graphviz import Source
    import plotly.graph_objects as go
    
    def clean_color(color):
        """Convert color to valid plotly format, handling alpha channels."""
        if not isinstance(color, str):
            return '#375a7f' if dark_mode else '#1f77b4'
            
        color = color.strip('"')
        
        if len(color) == 9 and color.startswith('#'):
            return color[:7]
            
        if color.startswith('rgba'):
            try:
                components = color.strip('rgba()').split(',')
                r, g, b = map(int, components[:3])
                return f'#{r:02x}{g:02x}{b:02x}'
            except:
                return '#375a7f' if dark_mode else '#1f77b4'
                
        if len(color) == 7 and color.startswith('#'):
            return color
            
        return '#375a7f' if dark_mode else '#1f77b4'
    
    # Read and parse DOT file
    try:
        graph = nx.drawing.nx_pydot.read_dot(dot_path)
    except Exception as e:
        print(f"Error reading DOT file: {e}")
        return None

    # Convert to undirected for layout calculation
    graph_undirected = graph.to_undirected()
    
    # Calculate layout using Kamada-Kawai with adjusted scale
    pos = nx.kamada_kawai_layout(graph_undirected, scale=2.0)
    
    # Extract node attributes
    node_x = []
    node_y = []
    node_labels = []
    node_colors = []
    hover_texts = []
    
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        attrs = graph.nodes[node]
        
        # Handle label with improved formatting
        #n0 [label="ID: 0\nSeq: GACGCGACTGTACGCTCACGACGACGGAGTCG\ncov: 43" style=filled fillcolor="#4895fa30"]
        label = attrs.get('label', node)
        if label and isinstance(label, str):
            label = label.strip('"')
            # Extract justrthe ID for node labels
            if 'ID:' in label:
                id_part = label.split('\\n')[0].replace('ID: ', '')
                seq_part = label.split('\\n')[1].replace('Seq: ', '')
                cov_part = label.split('\\n')[2].replace('Seq: ', '')
                label = f"Node {id_part} {seq_part} {cov_part}"
        node_labels.append("")
        
        # Create detailed hover text
        if isinstance(label, str) and 'Node' in label:
            hover_text = label.replace('\\n', '<br>')
            hover_texts.append(f"<b>{seq_part} {cov_part}</b>")
        else:
            hover_texts.append(f"")
        
        fillcolor = attrs.get('fillcolor', '#375a7f' if dark_mode else '#1f77b4')
        node_colors.append(clean_color(fillcolor))

    # Create edges with improved spacing
    edge_x = []
    edge_y = []
    edge_texts = []
    
    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        if line_shape == 'linear':
            # Direct edges without midpoint
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        else:
            # Curved edges with adjustable midpoint offset
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            offset = 0.05  # Adjust this value to control curve amount
            if abs(x1 - x0) > abs(y1 - y0):
                mid_y += offset
            else:
                mid_x += offset
            edge_x.extend([x0, mid_x, x1, None])
            edge_y.extend([y0, mid_y, y1, None])
        
        try:
            edge_data = graph.get_edge_data(u, v)
            label = edge_data.get('label', '') if edge_data else ''
            if isinstance(label, str):
                label = label.strip('"')
            edge_texts.append(f"<b>{u} → {v}</b><br>{label}" if label else f"<b>{u} → {v}</b>")
        except Exception as e:
            print(f"Error processing edge {u}-{v}: {e}")
            edge_texts.append(f"<b>{u} → {v}</b>")

    # Create figure with optimized layout
    fig = go.Figure()
    
    # Add edges with improved styling
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(
            width=1.5,
            color='#4f5b66' if dark_mode else '#888',
            shape=line_shape
        ),
        hoverinfo='text',
        hovertext=edge_texts,
        mode='lines',
        showlegend=False
    ))
    
    # Add nodes with improved styling
    node_marker = dict(
        showscale=False,
        color=node_colors,
        size=25,
        line_width=2,
        line_color='#4f5b66' if dark_mode else '#888',
        symbol='circle'
    )
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=hover_texts,
        text=node_labels,
        textposition="bottom center",
        textfont=dict(size=12),
        marker=node_marker,
        showlegend=False
    ))

    # Update layout with improved settings
    bg_color = '#2d3339' if dark_mode else '#ffffff'
    text_color = '#ffffff' if dark_mode else '#000000'
    
    # Calculate layout dimensions based on graph size
    x_range = max(node_x) - min(node_x)
    y_range = max(node_y) - min(node_y)
    
    # Add padding to ranges
    padding = 0.2
    x_min = min(node_x) - x_range * padding
    x_max = max(node_x) + x_range * padding
    y_min = min(node_y) - y_range * padding
    y_max = max(node_y) + y_range * padding
    
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        height=2000,  # Fixed height
        width=2000,  # Fixed width
        margin=dict(b=40, l=20, r=20, t=40),  # Adjusted margins
        annotations=[
            dict(
                text="Assembly Graph",
                showarrow=False,
                xref="paper", 
                yref="paper",
                x=0.5, 
                y=1.02,  # Moved title above plot
                font=dict(size=16, color=text_color)
            )
        ],
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[x_min, x_max],
            scaleanchor="y",  # This ensures equal scaling
            scaleratio=1      # Force 1:1 aspect ratio
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[y_min, y_max]
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font_color=text_color,
        hoverlabel=dict(
            bgcolor=bg_color,
            font_size=14,
            font_color=text_color
        )
    )
    
    return fig
#def 
# Define dark theme template for plotly
dark_template = dict(
    layout=dict(
        paper_bgcolor='#2d3339',  # Match Slate theme background
        plot_bgcolor='#2d3339',
        font=dict(color='#ffffff'),
        xaxis=dict(
            gridcolor='#4f5b66',
            linecolor='#4f5b66',
            zerolinecolor='#4f5b66'
        ),
        yaxis=dict(
            gridcolor='#4f5b66',
            linecolor='#4f5b66',
            zerolinecolor='#4f5b66'
        ),
        margin=dict(t=30, l=10, r=10, b=10)
    )
)

app_ui = ui.page_fluid(
    ui.h2("Contig Assembly Explorer"),
    
    # Layout with sidebar
    ui.layout_sidebar(
        # Sidebar panel
        ui.sidebar(
            ui.panel_well(
                ui.h3("Data Input"),
                ui.input_radio_buttons(
                    "input_type",
                    "Select Input Method",
                    {
                        "upload": "Upload File",
                        "remote": "Remote File Path"
                    },
                    selected="remote"
                ),
                ui.panel_conditional(
                    "input.input_type === 'remote'",
                    ui.input_text("remote_path", "Remote File Path", 
                                value='jijo.parquet',
                                placeholder="jijo.parquet")
                ),
                ui.panel_conditional(
                    "input.input_type === 'upload'",
                    ui.input_file("parquet_file", "Upload Parquet File", accept=[".parquet"])
                ),
            ),
            ui.panel_well(
                ui.h3("Assembly Parameters"),
                ui.input_selectize(
                    "umi", 
                    "Select UMI",
                    choices=[]
                ),
                ui.input_slider(
                    "kmer_size",
                    "K-mer Size",
                    min=15,
                    max=63,
                    value=31,
                    step=2
                ),
                ui.input_slider(
                    "min_coverage",
                    "Minimum Coverage",
                    min=1,
                    max=50,
                    value=17
                ),
                ui.input_checkbox(
                    "auto_k",
                    "Auto K-mer Size",
                    value=True
                ),
                ui.input_action_button(
                    "assemble",
                    "Assemble Contig",
                    class_="btn-primary"
                ),
                ui.hr(),  # Add a visual separator
                ui.hr(),  # Add a visual separator
                ui.hr(),  # Add a visual separator
                ui.h4("Parameter Sweep"),
                ui.input_slider(
                    "k_start",
                    "K-mer Range Start",
                    min=1,
                    max=20,
                    value=1,
                    step=1
                ),

                ui.input_slider(
                    "k_end",
                    "K-mer Range End",
                    min=21,
                    max=150,
                    value=75,
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
                    min=250,
                    max=500,
                    value=100
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
            ),
            width=500,
        ),
        
        # Main panel
        ui.navset_tab(
            ui.nav_panel("Overview",
                ui.row(
                    ui.column(6,
                        ui.panel_well(
                            ui.h4("Summary Statistics"),
                            ui.p("Total UMIs: ", ui.output_text("total_umis")),
                            ui.p("Median Reads/UMI: ", ui.output_text("median_reads"))
                        )
                    ),
                    ui.column(6,
                        ui.panel_well(
                            ui.h4("Selected UMI Stats"),
                            ui.output_text("selected_umi_stats")
                        )
                    )
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
                    ui.column(6,
                        ui.panel_well(
                            ui.h4("Assembly Statistics"),
                            ui.pre(ui.output_text("assembly_stats"))
                        )
                    ),
                    ui.column(6,
                        ui.panel_well(
                            ui.h4("Contig Sequence"),
                            ui.output_text("contig_graph_path"),
                            ui.output_text_verbatim("contig_sequence")
                        )
                    )
                ),
            ui.row(
                ui.column(3,
                    ui.panel_well(
                        ui.h4("Graph Visualization Options"),
                        ui.input_select(
                            "graph_type",
                            "Graph Type",
                            choices={
                                "compressed": "Compressed Graph",
                                "preliminary": "Preliminary Graph"
                            },
                            selected="compressed"
                        ),
                        ui.input_select(
                            "line_shape",
                            "Edge Style",
                            choices={
                                "linear": "Straight",
                                "spline": "Curved",
                                "vh": "Vertical-Horizontal Steps",
                                "hv": "Horizontal-Vertical Steps"
                            },
                            selected="linear"
                        ),
                    )
                )
            ),
                         ui.row(
                        ui.h4("Assembly Graph"),
                        output_widget("assembly_graph")

                             ),

            ),
            ui.nav_panel("Parameter Sweep",
                ui.row(
                    ui.column(6,
                        ui.panel_well(
                            ui.h4("K-mer vs Coverage Sweep"),
                            output_widget("sweep_heatmap")
                        )
                    )
                )
            )
        )
    ),
    theme=shinyswatch.theme.slate()
)

def server(input, output, session):
    # Reactive data storage
    data = reactive.Value(None)
    assembly_result = reactive.Value(None)
    sweep_results = reactive.Value(None)
    
    @render.text
    def value():
        return input.parquet_file()

    @reactive.Effect
    @reactive.event(input.parquet_file, input.remote_path, input.input_type)
    def load_data():
        try:
            if input.input_type() == "upload" and input.parquet_file() is not None:
                file_path = input.parquet_file()[0]["datapath"]
            elif input.input_type() == "remote" and input.remote_path:
                file_path = input.remote_path()
            else:
                return
                
            if not Path(file_path).exists():
                ui.notification_show(
                    f"File not found: {file_path}",
                    type="error"
                )
                return
            ui.notification_show(f"loading from {input.input_type()}")     
            ui.notification_show(f"loading from {file_path}")     
            df = pl.read_parquet(file_path)
            data.set(df)
            
            # Update UMI choices
            umis = df.select('umi').unique().to_series().to_list()
            ui.update_selectize(
                "umi",
                choices=umis,
                selected=umis[0] if umis else None
            )
            
        except Exception as e:
            ui.notification_show(
                f"Error loading data: {str(e)}",
                type="error"
            )
    
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
    def selected_umi_stats():
        if data() is None or not input.umi():
            return "No UMI selected"
        
        df = data()
        umi_reads = df.filter(pl.col('umi') == input.umi()).height
        
        return f"""
        Selected UMI: {input.umi()}\n
        Number of reads: {umi_reads}
        """

    @output
    @render_plotly
    def coverage_dist():
        if data() is None:
            return go.Figure()
                
        reads_per_umi = (
            data()
            .group_by('umi')
            .agg(pl.count().alias('reads'))
        )

        fig = px.histogram(
            reads_per_umi.to_pandas(),
            x='reads',
            title="Distribution of Reads per UMI",
            labels={'reads': 'Number of Reads'},
            nbins=50,
            template='plotly_dark'
        )
        
        fig.update_layout(
            **dark_template['layout'],
            showlegend=False,
            xaxis_title="Reads per UMI",
            yaxis_title="Count",
            bargap=0.1
        )
        
        # Update bars color to match theme
        fig.update_traces(marker_color='#375a7f')  # Blue shade matching Slate theme
        
        return fig

    @output
    @render_plotly
    def reads_per_umi():
        if data() is None:
            return go.Figure()
            
        reads_per_umi = (
            data()
            .group_by('umi')
            .agg(pl.count().alias('reads'))
            .sort('reads', descending=True)
            .head(50)
        )
        
        fig = px.bar(
            reads_per_umi.to_pandas(),
            x='umi',
            y='reads',
            title="Top 50 UMIs by Read Count",
            template='plotly_dark'
        )
        
        fig.update_layout(
            **dark_template['layout'],
            xaxis_tickangle=45,
            showlegend=False,
            xaxis_title="UMI",
            yaxis_title="Read Count"
        )
        
        # Update bars color to match theme
        fig.update_traces(marker_color='#375a7f')  # Blue shade matching Slate theme
        
        return fig

    @reactive.Effect
    @reactive.event(input.assemble)
    def run_assembly():
        try:
            if data() is None or not input.umi():
                ui.notification_show(
                    "Please load data and select a UMI first",
                    type="warning"
                )
                return
                
            df = data()
            result = (
                df
                .pp.assemble_umi(
                    target_umi=input.umi(),
                    k=input.kmer_size(),
                    min_cov=input.min_coverage(),
                    auto_k=input.auto_k(),
                    export_graphs=True,
                    only_largest=True,
                    intbc_5prime='GAGACTGCATGG'
                )
            )
            
            if len(result) > 0:
                contig = result.get_column('contig')[0]
                assembly_result.set(contig)
                ui.notification_show(
                    f"Assembly completed successfully! Contig length: {len(contig)}",
                    type="message"
                )
            else:
                ui.notification_show(
                    "Assembly produced no contigs",
                    type="warning"
                )
                assembly_result.set(None)
                
        except Exception as e:
            ui.notification_show(
                f"Error during assembly: {str(e)}",
                type="error"
            )
            print(f"Assembly error: {str(e)}")

    @output
    @render.text
    def assembly_stats():
        if data() is None or assembly_result() is None:
            return "No assembly results available"
            
        read_count = data().filter(pl.col('umi') == input.umi()).height
        return f"""Assembly Statistics
        Contig Length: {len(assembly_result())}
        Input Reads: {read_count}
        Selected UMI: {input.umi()}
        K-mer Size: {input.kmer_size()}
        Min Coverage: {input.min_coverage()}"""


    @output
    @render.text
    def contig_graph_path():
        if assembly_result() is None:
            return "No contig sequence available"
        graph_path = f"{Path(__file__).parent}/{input.umi()}__preliminary.dot"
        return graph_path 

    @output
    @render.text
    def contig_sequence():
        if assembly_result() is None:
            return "No contig sequence available"
        return assembly_result()


    # Add to existing server function:

    @reactive.Effect
    @reactive.event(input.run_sweep)
    def run_parameter_sweep():
        try:
            if data() is None or not input.umi():
                ui.notification_show(
                    "Please load data and select a UMI first",
                    type="warning"
                )
                return
                
            df = data()
            filtered_df = df.filter(pl.col('umi') == input.umi())
            
            ui.notification_show(
                "Running parameter sweep...",
                type="message"
            )
            
            # Run sweep using the API
            result = (
                filtered_df
                .pp.assemble_sweep_params_umi(
                    target_umi=input.umi(),
                    k_start=input.k_start(),
                    k_end=input.k_end(),
                    k_step=input.k_step(),
                    cov_start=input.cov_start(),
                    cov_end=input.cov_end(),
                    cov_step=5,
                    export_graphs=False,
                    intbc_5prime='GAGACTGCATGG',
                    prefix=input.umi()
                )
                .drop('umi')
                .pivot(index='k', on='min_coverage', values='contig_length')
            )

            #result.write_parquet(f'{input.umi()}_sweep.parquet')
            result = result.to_pandas().set_index('k')
            
            sweep_results.set(result)
            
            ui.notification_show(
                "Parameter sweep completed!",
                type="message"
            )
            
        except Exception as e:
            ui.notification_show(
                f"Error during parameter sweep: {str(e)}",
                type="error"
            )
            print(f"Sweep error: {str(e)}")

    @output
    @render_plotly
    def sweep_heatmap():
        if sweep_results() is None:
            return go.Figure(layout=dark_template['layout'])
            
        # Create heatmap using plotly express
        fig = px.imshow(
            sweep_results(),
            labels=dict(
                x="Minimum Coverage",
                y="K-mer Size",
                color="Contig Length"
            ),
            title="K-mer Size vs Coverage Parameter Sweep",
            aspect="auto",  # Maintain readability regardless of dimensions
            color_continuous_scale="Viridis"  # Use a color scale that works well with dark theme
        )
        
        # Update layout to match dark theme
        fig.update_layout(
            **dark_template['layout'],
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
    def assembly_graph():
        if assembly_result() is None:
            return go.Figure(layout=dark_template['layout'])
            
        # Build path based on selected graph type
        base_path = f"{Path(__file__).parent}/{input.umi()}"
        graph_path = f"{base_path}__{input.graph_type()}.dot"
        
        if not Path(graph_path).exists():
            return go.Figure(layout=dark_template['layout'])
            
        # Create graph visualization with selected options
        fig = create_graph_plot(
            graph_path, 
            dark_mode=True,
            line_shape=input.line_shape(),
            graph_type=input.graph_type()
        )
        
        if fig is None:
            # Return empty figure with message if graph creation failed
            fig = go.Figure()
            fig.update_layout(
                **dark_template['layout'],
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
        return fig

app = App(app_ui, server)
