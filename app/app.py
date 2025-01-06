from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_plotly
import shinyswatch
import polars as pl
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from ogtk.ltr.fracture.pipeline import api_ext

def sync_slider_and_text(input, session, slider_id, text_id, shared_value):
    # Update shared_value when slider changes
    @reactive.Effect
    @reactive.event(input[slider_id])
    def _():
        shared_value.set(input[slider_id]())
        # Update text input to match slider value
        ui.update_text(text_id, value=str(input[slider_id]()))

    # Update shared_value when text input changes
    @reactive.Effect
    @reactive.event(input[text_id])
    def _():
        try:
            # Convert text to integer and update shared_value
            new_value = int(input[text_id]())
            shared_value.set(new_value)
            # Update slider to match text input value
            ui.update_slider(slider_id, value=new_value)
        except ValueError:
            pass  # Ignore invalid input


def split_and_with_spacer(s, n, spacer='<br>'):
    return spacer.join([s[i:i+n] for i in range(0, len(s), n)])

def clean_color(color, dark_mode=True):
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


def split_and_with_spacer(s, n, spacer='<br>'):
    """Split a string into chunks of size n with a spacer between them."""
    return spacer.join([s[i:i+n] for i in range(0, len(s), n)]) 

def highlight_sequences(text, sequence_colors):
    """Highlight specific sequences in the hover text with proper HTML escaping."""
    import html
    
    # First escape any HTML in the original text
    escaped_text = html.escape(text)
    result = escaped_text
    
    # Keep track of where we've inserted spans to avoid nested tags
    # Each entry will be (start_pos, end_pos, html_to_insert)
    replacements = []
    
    # Find all occurrences of each sequence and their positions
    for seq, color in sequence_colors.items():
        start = 0
        while True:
            pos = escaped_text.find(seq, start)
            if pos == -1:
                break
            replacements.append((
                pos, 
                pos + len(seq), 
                f'<span style="color: {color}">{seq}</span>'
            ))
            start = pos + 1
    
    # Sort replacements by start position in reverse order
    replacements.sort(key=lambda x: x[0], reverse=True)
    
    # Apply the replacements
    for start, end, html in replacements:
        result = result[:start] + html + result[end:]
    
    return result

def format_sequence_with_highlights(seq_part, sequence_colors, line_length=60):
    """Format sequence with highlights, handling sequences that span line breaks."""
    # First highlight the entire sequence
    highlighted_seq = highlight_sequences(seq_part, sequence_colors)
    
    # Now split the highlighted sequence into lines, being careful with HTML tags
    lines = []
    current_line = []
    char_count = 0
    in_tag = False
    tag_buffer = []
    
    for char in highlighted_seq:
        if char == '<':
            in_tag = True
            tag_buffer.append(char)
            continue
            
        if in_tag:
            tag_buffer.append(char)
            if char == '>':
                in_tag = False
                current_line.append(''.join(tag_buffer))
                tag_buffer = []
            continue
        
        current_line.append(char)
        char_count += 1
        
        if char_count >= line_length:
            lines.append(''.join(current_line))
            current_line = []
            char_count = 0
    
    if current_line:
        lines.append(''.join(current_line))
        
    if tag_buffer:  # Handle any unclosed tags (shouldn't happen with valid HTML)
        lines[-1] += ''.join(tag_buffer)
        
    return '<br>'.join(lines)

def get_node_style(seq, sequence_colors, dark_mode):
    """Determine node color and opacity based on sequence presence."""
    if not seq:
        return dict(color='rgba(0,0,0,0)', line=dict(color='#4f5b66' if dark_mode else '#888', width=2))
    
    # Clean up the sequence string
    seq = seq.strip('"').strip()
    
    for target_seq, color in sequence_colors.items():
        if target_seq in seq:
            return dict(color=color, line=dict(color='#4f5b66' if dark_mode else '#888', width=2))
    
    # No matching sequence found - return transparent fill with border
    return dict(color='rgba(0,0,0,0)', line=dict(color='#4f5b66' if dark_mode else '#888', width=2))

def extract_coverage(label):
    """Extract coverage value from node label."""
    if not label or not isinstance(label, str):
        return None
    import re 
    match = re.search(r'cov:\s*(\d+)', label)
    if match:
        return int(match.group(1))
    return None


def create_graph_plot(dot_path, dark_mode=True, line_shape='linear', graph_type='compressed', debug=False):
    """Convert a DOT file to a Plotly figure with optimized layout settings.
    Nodes are sized according to their coverage values and disjoint subgraphs are separated.
    
    Args:
        dot_path (str): Path to the DOT file
        dark_mode (bool): Whether to use dark theme colors
        line_shape (str): Shape of edges ('linear' or 'spline')
        graph_type (str): Type of graph visualization
        debug (bool): Whether to print debug information
        
    Returns:
        go.Figure: Plotly figure object
    """
    import networkx as nx
    from graphviz import Source
    import plotly.graph_objects as go
    import re
    import html
    from collections import defaultdict
    
    # Read and parse DOT file
    try:
        graph = nx.drawing.nx_pydot.read_dot(dot_path)
    except Exception as e:
        print(f"Error reading DOT file: {e}")
        return None

    # Find connected components (disjoint subgraphs)
    components = list(nx.connected_components(graph.to_undirected()))
    
    # Calculate layout for each component separately and then adjust positions
    all_pos = {}
    component_centers = []
    spacing = 3.0  # Increase spacing between components
    
    for i, component in enumerate(components):
        # Create subgraph for this component
        subgraph = graph.subgraph(component)
        
        # Calculate layout for this component
        subgraph_pos = nx.kamada_kawai_layout(subgraph.to_undirected(), scale=2.0)
        
        # Find center of this component
        center_x = sum(pos[0] for pos in subgraph_pos.values()) / len(subgraph_pos)
        center_y = sum(pos[1] for pos in subgraph_pos.values()) / len(subgraph_pos)
        component_centers.append((center_x, center_y))
        
        # Store positions for this component's nodes
        all_pos.update(subgraph_pos)

    # Adjust component positions to prevent overlap
    if len(components) > 1:
        for i, component in enumerate(components):
            # Calculate offset for this component
            offset_x = (i % 2) * spacing
            offset_y = (i // 2) * spacing
            
            # Apply offset to all nodes in this component
            for node in component:
                all_pos[node] = (
                    all_pos[node][0] + offset_x,
                    all_pos[node][1] + offset_y
                )
    
    pos = all_pos  # Use the adjusted positions
    
    # Extract node attributes
    node_x = []
    node_y = []
    node_labels = []
    node_colors = []
    node_sizes = []
    hover_texts = []
    
    # Track min/max coverage for scaling
    coverages = []
    
    # Define sequences and their corresponding colors
    sequence_colors = {
        'GAGACTGCATGG': '#50C878',  # Emerald green for theme
        'TTTAGTGAGGGT': '#9370DB'   # Medium purple for theme
    }
    
    # First pass to collect coverage values
    for node in graph.nodes():
        attrs = graph.nodes[node]
        label = attrs.get('label', '')
        if isinstance(label, str):
            coverage = extract_coverage(label.strip('"'))
            if coverage is not None:
                coverages.append(coverage)
    
    # Calculate size scaling factors
    min_coverage = min(coverages) if coverages else 1
    max_coverage = max(coverages) if coverages else 1
    min_size = 10
    max_size = 60
    
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        attrs = graph.nodes[node]
        label = attrs.get('label', node)
        
        if isinstance(label, str):
            # Handle both escaped and unescaped newlines
            label = label.strip('"').replace('\\\\n', '\n').replace('\\n', '\n')
            parts = label.split('\n')
            
            # Extract ID, sequence, and coverage
            id_part = ''
            seq_part = ''
            cov_part = ''
            
            for part in parts:
                if 'ID:' in part:
                    id_part = part.replace('ID:', '').strip()
                elif 'Seq:' in part:
                    seq_part = part.replace('Seq:', '').strip()
                elif 'cov:' in part:
                    cov_part = part.replace('cov:', '').strip()
            
            if id_part and seq_part:
                label = ""
                
                # Create hover text with proper indentation
                hover_text = []
                if id_part:
                    hover_text.append(f"<b>ID:</b> {html.escape(id_part)}")
                if seq_part:
                    # Use the formatter that handles cross-line highlights
                    highlighted_seq = format_sequence_with_highlights(seq_part, sequence_colors)
                    hover_text.append(f"<b>Sequence:</b><br>{highlighted_seq}")
                if cov_part:
                    hover_text.append(f"<b>Coverage:</b> {html.escape(cov_part)}")
                
                hover_texts.append('<br>'.join(hover_text))
                
                # Get node style based on sequence
                node_style = get_node_style(seq_part, sequence_colors, dark_mode)
                node_colors.append(node_style['color'])
            else:
                label = str(node)
                hover_texts.append(f"Node: {node}")
                node_style = get_node_style(None, sequence_colors, dark_mode)
                node_colors.append(node_style['color'])
        else:
            label = str(node)
            hover_texts.append(f"Node: {node}")
            node_style = get_node_style(None, sequence_colors, dark_mode)
            node_colors.append(node_style['color'])
        
        node_labels.append(label)
        
        # Calculate node size based on coverage
        coverage = extract_coverage(attrs.get('label', ''))
        if coverage is not None and max_coverage > min_coverage:
            size = min_size + (max_size - min_size) * (coverage - min_coverage) / (max_coverage - min_coverage)
        else:
            size = min_size
        node_sizes.append(size)

    # Create edges
    edge_x = []
    edge_y = []
    edge_texts = []
    
    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        if line_shape == 'linear':
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        else:
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            offset = 0.05
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

    # Create figure
    fig = go.Figure()
    
    # Add edges
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
    
    # Add nodes
    node_marker = dict(
        showscale=False,
        color=node_colors,
        size=node_sizes,
        line=dict(
            width=2,
            color='#4f5b66' if dark_mode else '#888'
        ),
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

    # Add legend for sequence colors
    for seq, color in sequence_colors.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            showlegend=True,
            name=f'Sequence: {seq}'
        ))

    # Layout settings
    bg_color = '#2d3339' if dark_mode else '#ffffff'
    text_color = '#ffffff' if dark_mode else '#000000'
    
    # Calculate padding based on component spread
    x_range = max(node_x) - min(node_x)
    y_range = max(node_y) - min(node_y)
    
    padding = 0.3  # Increased padding to accommodate separated components
    x_min = min(node_x) - x_range * padding
    x_max = max(node_x) + x_range * padding
    y_min = min(node_y) - y_range * padding
    y_max = max(node_y) + y_range * padding
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor=bg_color,
            bordercolor=text_color,
            borderwidth=1,
            font=dict(color=text_color)
        ),
        hovermode='closest',
        height=1000,
        width=1000,
        margin=dict(b=40, l=20, r=20, t=40),
        annotations=[
            dict(
                text="Assembly Graph",
                showarrow=False,
                xref="paper", 
                yref="paper",
                x=0.5, 
                y=1.02,
                font=dict(size=16, color=text_color)
            )
        ],
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[x_min, x_max],
            scaleanchor="y",
            scaleratio=1
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

                ui.h3("Parameter Sweep"),
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
                ui.hr(),
                ui.hr(),
                ui.hr(),

                ui.h3("Assembly Parameters"),
                ui.input_selectize(
                    "umi", 
                    "Select UMI",
                    choices=[]
                ),
                # ui.input_slider(
                #     "min_coverage",
                #     "Minimum Coverage",
                #     min=1,
                #     max=200,
                #     value=17
                # ),
                ui.input_text(
                     "min_coverage",
                     "Minimum Coverage",
                     value=17,
                    placeholder=17),
                ui.input_text(
                    "kmer_size", 
                    "K-mer Range Start", 
                    value=1,
                    placeholder=1),

                # ui.input_slider(
                #     "kmer_size",
                #     "K-mer Size",
                #     min=15,
                #     max=63,
                #     value=31,
                #     step=2
                # ),
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
                ui.hr(),
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
                        ui.panel_well(
                            ui.output_ui("assembly_stats"),
                            ui.h3("Contig Sequence"),
                            ui.output_text_verbatim("contig_sequence"),

                        )
                    ),
                    ui.row(
                        ui.column(3,
                            ui.panel_well(
                                ui.h4("K-mer vs Coverage Sweep"),
                                output_widget("sweep_heatmap")
                            )
                        ),
                        ui.column(9,
                            ui.h4("Assembly Graph"),
                            output_widget("assembly_graph")
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

                             ),

            ),
        )
    ),
    theme=shinyswatch.theme.slate()
)

def server(input, output, session):
    # TODO: example of how to sync inputs
    #     # Create reactive variables for shared values
    shared_value1 = reactive.Value(None)
    # shared_value2 = reactive.Value(1800)
    #
    # # Use the helper function for each pair of inputs
    # sync_slider_and_text(input, session, "kmer_size", "kmer_size_txt", shared_value1)
    # sync_slider_and_text(input, session, "slider2", "text2", shared_value2)
    #

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
                    k=int(input.kmer_size()),
                    min_cov=int(input.min_coverage()),
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
        return ui.HTML(f"""
    Target UMI: <h4>{input.umi()}</h4>
    Contig Length: <span style="color: #ff8080;">{len(assembly_result())}</span> 
    Input Reads: <span style="color: #ffffff;">{read_count}</span> 
    K-mer Size: <span style="color: #ffa07a;">{input.kmer_size()}</span> 
    Min Coverage: <span style="color: #ffa07a;">{input.min_coverage()}</span>"""
        )

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
