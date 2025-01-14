from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_plotly
from shiny.session import get_current_session
import shinyswatch


import polars as pl
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
#from psutil import cpu_count, cpu_percent
import numpy as np


import tempfile
import os
from pathlib import Path
import html

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")

import ogtk.ltr.align as al

# max number of samples to retain
MAX_SAMPLES = 1000
# secs between samples
SAMPLE_PERIOD = 1

SYSTEM_PREFIXES = {
        "wexac": "/home/pedro/wexac/",
        "labs": "/home/labs/nyosef/pedro/",
        "laptop": "/Volumes/pedro",
        }

db = (
        pl.read_csv('../parquet_db.txt', has_header=False)
        .rename({'column_1':'full_path'})
        .with_columns(pl.col('full_path').str.split("/").alias('split'))
        .with_columns(name=pl.col('split').list.get(-3) +"::"+pl.col('split').list.get(-2))
        .drop('split')
        .sort('name')
        )

db = {i:ii for i,ii in db.iter_rows()}

def highlight_sequences_in_table(text):
    """Highlight specific sequences in text with HTML formatting."""
    sequence_colors = {
            'GAGACTGCATGG': '#50C878',  # Emerald green for theme
            'TTTAGTGAGGGT': '#9370DB'   # Medium purple for theme
            }

    # First escape any HTML in the original text
    escaped_text = html.escape(text)
    result = escaped_text

    # Keep track of where we've inserted spans to avoid nested tags
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
    for start, end, html_span in replacements:
        result = result[:start] + html_span + result[end:]

    return result

def format_top_contigs_table(df):
    """Format top contigs dataframe as an HTML table with highlighted sequences."""
    html_output = ["<table class='table table-dark table-striped'>"]

    # Add header
    html_output.append("<thead><tr>")
    for col in ['Node ID', 'Sequence', 'Coverage', 'Length']:
        html_output.append(f"<th>{col}</th>")
    html_output.append("</tr></thead>")

    # Add body
    html_output.append("<tbody>")
    for row in df.iter_rows():
        html_output.append("<tr>")
        # Node ID
        html_output.append(f"<td>{row[0]}</td>")
        # Sequence with highlighting
        highlighted_seq = highlight_sequences_in_table(row[1])
        html_output.append(f"<td style='font-family: monospace;'>{highlighted_seq}</td>")
        # Coverage
        html_output.append(f"<td>{row[2]}</td>")
        # Length
        html_output.append(f"<td>{row[3]}</td>")
        html_output.append("</tr>")

    html_output.append("</tbody></table>")
    return "\n".join(html_output)



from ogtk.ltr.fracture.pipeline import api_ext

pl.Config().set_fmt_str_lengths(666)




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

def get_node_style(seq, sequence_colors, dark_mode, path_nodes=None):
    """Determine node color and opacity based on sequence presence."""
    if not seq:
        return dict(color='rgba(0,0,0,0)', line=dict(color='#4f5b66' if dark_mode else '#888', width=2))

    # Clean up the sequence string
    seq = seq.strip('"').strip()

    # Check if node is in path but doesn't contain anchor sequences
    if path_nodes is not None:
        node_id = None
        for part in seq.split('\n'):
            if 'ID:' in part:
                try:
                    node_id = int(part.replace('ID:', '').strip())
                    break
                except ValueError:
                    pass
        
        if node_id in path_nodes and not any(target_seq in seq for target_seq in sequence_colors.keys()):
            return dict(color='#FFD700', line=dict(color='#4f5b66' if dark_mode else '#888', width=2))  # Yellow

    has_both = all(target_seq in seq for target_seq in sequence_colors.keys())
    if has_both:
        return dict(color='#FFA500', line=dict(color='#4f5b66' if dark_mode else '#888', width=2))

    for target_seq, color in sequence_colors.items():
        if target_seq in seq:
            return dict(color=color, line=dict(color='#4f5b66' if dark_mode else '#888', width=2))

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


def create_graph_plot(dot_path, dark_mode=True, line_shape='linear', graph_type='compressed', debug=False, path_nodes=None):
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
    path_file = dot_path.replace('.dot', '_path.dot')

    if os.path.exists(path_file):
        try:
            path_graph = nx.drawing.nx_pydot.read_dot(path_file)
            path_nodes = set(path_graph.nodes())
        except Exception as e:
            print(f"Error reading path file: {e}")

    try:
        graph = nx.drawing.nx_pydot.read_dot(dot_path)

        if len(graph.nodes()) == 0:
            print(f"Loaded .dot file has no nodes {dot_path}")
            fig = go.Figure()
            fig.update_layout(
                    **dark_template['layout'],
                    #width=1000,
                    height=1000,
                    autosize=True,
                    annotations=[dict(
                        text="Empty graph - no nodes found",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(color='#ffffff', size=14)
                        )]
                    )
            return fig

    except Exception as e:
        print(f"Error in create_graph_plot: {str(e)}")
        fig = go.Figure()
        fig.update_layout(
                **dark_template['layout'],
                # width=1000,
                height=1000,
                autosize=True,
                annotations=[dict(
                    text=f"Error creating graph: {str(e)}",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color='#ffffff', size=14)
                    )]
                )
        return fig


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
                node_style = get_node_style(seq_part, sequence_colors, dark_mode, path_nodes=path_nodes)
                node_colors.append(node_style['color'])
            else:
                label = str(node)
                hover_texts.append(f"Node: {node}")
                node_style = get_node_style(seq_part, sequence_colors, dark_mode, path_nodes=path_nodes)
                node_colors.append(node_style['color'])
        else:
            label = str(node)
            hover_texts.append(f"Node: {node}")
            node_style = get_node_style(seq_part, sequence_colors, dark_mode, path_nodes=path_nodes)
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

    # # Add legend for sequence colors
    # for seq, color in sequence_colors.items():
    #     fig.add_trace(go.Scatter(
    #         x=[None],
    #         y=[None],
    #         mode='markers',
    #         marker=dict(size=10, color=color),
    #         showlegend=True,
    #         name=f'Sequence: {seq}'
    #         ))

    # Layout settings
    bg_color = '#2d3339' if dark_mode else '#ffffff'
    text_color = '#ffffff' if dark_mode else '#000000'

    # Calculate padding based on component spread
    x_range = max(node_x) - min(node_x)
    y_range = max(node_y) - min(node_y)

    padding = 0.02
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
            # width=1000,
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
        ui.tags.style("""
        #shiny-notification-panel {
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 1050; /* Ensure it appears on top */
        }
    """),
        ui.h2("Contig Assembly Explorer"),

        # Layout with sidebar
        ui.layout_sidebar(
            # Sidebar panel
            ui.sidebar(
                ui.panel_well(
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
                        selected="remote"
                        ),
                    ui.panel_conditional(
                        "input.input_type === 'remote'",

                        ui.input_selectize(
                            "system_prefix",
                            "Select System",
                            choices=SYSTEM_PREFIXES,
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
                ui.panel_well(
                        ui.hr(),
                        ),
                # ui.accordion(
                #     ui.accordion_panel("CPU Monitor",
                #                        ui.input_switch("cpu_hold", "Freeze Display", value=True),
                #                        ui.input_action_button("cpu_reset", "Clear History", class_="btn-sm"),
                #                        ui.input_numeric("sample_count", "Samples to show", value=50),
                #                        ui.output_plot("cpu_plot"),
                #                        ),
                #     open=False,
                #
                #     ),
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
                                                   ui.output_ui("assembly_stats"),
                                                   ui.h3("Contig Sequence"),
                                                   ui.output_text_verbatim("contig_sequence"),
                                                   ui.output_ui("top_contigs"),
                                                   ui.h3("Path Results"),
                                                   ui.output_ui("path_results_output"),
                                                   ui.input_text("top_path", "Top n paths", value=10, placeholder="How many paths to display (sorted byt coverage)"),
                                                   ),
                                               ),
                                     ),
                                 ui.row(
                                     ui.column(9,
                                               ui.panel_well("Polished Contigs Stats",
                                                             ui.h3("Contig Sequence"),
                                                             ui.output_ui("top_contigs_polish"),
                                                             ),
                                               open=False,
                                               ),
                                     ),
                                 ui.row(
                                     ui.column(4,
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
                                                       max=64,
                                                       value=64,
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
                                        ),
                                           ),
ui.column(4,
          ui.panel_well(

              ui.input_radio_buttons(
                  "assembly_method",
                  "Select Assembly Method",
                  {
                      "compression": "Graph Compression",
                      "shortest_path": "Shortest Path"
                      },
                  selected="shortest_path"
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
                  value=17,
                  ),
              ui.input_numeric(
                  "kmer_size", 
                  "K-mer Size", 
                  value=17,
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
                  )
              )
          ),
ui.column(4,
          ui.panel_well(
              ui.h4("Polish Assembly Parameters"),
              ui.input_numeric(
                  "polish_min_coverage",
                  "Polish Minimum Coverage",
                  value=1,
                  ),
              ui.input_numeric(
                  "polish_kmer_size", 
                  "Polish K-mer Size", 
                  value=8,
                  ),
              ui.input_checkbox(
                  "polish_auto_k",
                  "Auto K-mer Size",
                  value=False
                  ),
              ui.input_action_button(
                  "polish",
                  "Polish Assembly",
                  class_="btn-primary"
                  ),
              )
          ),
),
                    ui.row(
                            ui.column(4,
                                      ui.panel_well(
                                          ui.h4("K-mer vs Coverage Sweep"),
                                          ui.div(output_widget("sweep_heatmap"), style="height: 500px;"),
                                          )
                                      ),
                            ui.column(4,
                                      ui.panel_well(
                                          ui.h4("Assembly Graph"),
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
                                              ui.input_switch(
                                                  "use_polished",
                                                  "Use Polished Assembly",
                                                  value=False
                                                  ),
                                              ),
                                          # output_widget("assembly_graph")
                                          ui.div(output_widget("assembly_graph"), style="min-height: 1000px; height: auto; width: 100%")
                                          )

                                      ),
                            ui.column(4,
                                      ui.h4("Coverage Plot (from uniqued reads!!)"),
                                      ui.div(output_widget("coverage_plot"), style="height: 500px;"),
                                      ),
                            ),

            ),

            ui.nav_panel("Sweep Results",
                         ),
            selected="Assembly Results"  
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
    path_results = reactive.Value(None)

    #ncpu = cpu_count(logical=True)
    #cpu_history = reactive.Value(None)

    #@reactive.calc
    #def cpu_current():
    #    reactive.invalidate_later(SAMPLE_PERIOD)
    #    return cpu_percent(percpu=True)

    #@reactive.calc
    #def cpu_history_with_hold():
    #    if not input.cpu_hold():
    #        return cpu_history()
    #    else:
    #        input.cpu_reset()
    #        with reactive.isolate():
    #            return cpu_history()

    #@reactive.effect
    #def collect_cpu_samples():
    #    new_data = np.vstack(cpu_current())
    #    with reactive.isolate():
    #        if cpu_history() is None:
    #            cpu_history.set(new_data)
    #        else:
    #            combined_data = np.hstack([cpu_history(), new_data])
    #            if combined_data.shape[1] > MAX_SAMPLES:
    #                combined_data = combined_data[:, -MAX_SAMPLES:]
    #            cpu_history.set(combined_data)

    #@reactive.effect(priority=100)
    #@reactive.event(input.cpu_reset)
    #def reset_cpu_history():
    #    cpu_history.set(None)

    #def plot_cpu(history, samples, ncpu, cmap='viridis'):
    #    """Plot CPU history using matplotlib."""
    #    if history is None or history.shape[1] == 0:
    #        fig, ax = plt.subplots(figsize=(8, 4))
    #        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    #        ax.set_xticks([])
    #        ax.set_yticks([])
    #        return fig

    #    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), height_ratios=[3, 1])
    #    fig.patch.set_facecolor("#2d3339")

    #    # Last n samples for the line plots
    #    n = min(samples, history.shape[1])
    #    x = np.arange(n)

    #    # Line plot for each CPU
    #    for i in range(ncpu):
    #        y = history[i, -n:]
    #        ax1.plot(x, y, alpha=0.7, linewidth=1)

    #    ax1.set_xlim(-0.5, n - 0.5)  # Add 0.5 unit buffer on each side
    #    ax1.set_ylim(0, 100)
    #    ax1.set_ylabel("CPU %")
    #    ax1.grid(True, alpha=0.3)
    #    ax1.set_facecolor("#2d3339")
    #    ax1.tick_params(colors="white")
    #    for spine in ax1.spines.values():
    #        spine.set_color("white")

    #    # Heatmap
    #    if history.shape[1] > 0:
    #        ax2.imshow(
    #                history[:, -n:],
    #                aspect="auto",
    #                cmap=cmap,
    #                interpolation="nearest",
    #                vmin=0,
    #                vmax=100,
    #                )

    #    ax2.set_yticks(range(ncpu))
    #    ax2.set_yticklabels([f"CPU {i}" for i in range(ncpu)])
    #    ax2.tick_params(colors="white")

    #    fig.tight_layout()
    #    return fig

    #@render.plot
    #def cpu_plot():
    #    """Render the CPU plot."""
    #    history = cpu_history_with_hold()
    #    if history is None:
    #        return plot_cpu(None, input.sample_count(), ncpu)
    #    return plot_cpu(history, input.sample_count(), ncpu)


    @render.text
    def value():
        return input.parquet_file()

    @reactive.Effect
    @reactive.event(input.parquet_file_local, input.parquet_file, input.remote_path, input.input_type, input.system_prefix, input.sample_n_umis, input.sample_min_reads, input.sample_umis)

    def load_data():
        with reactive.isolate():
            data.set(None)
            sweep_results.set(None) 
            assembly_result.set(None)
        ui.update_selectize("umi", choices=[], selected=None)

        try:
            if input.input_type() == "upload" and input.parquet_file() is not None:
                file_path = input.parquet_file()[0]["datapath"]
            elif input.input_type() == "local" and input.parquet_file_local() is not None:
                file_path = input.parquet_file_local()
            elif input.input_type() == "remote" and input.remote_path:
                prefix = SYSTEM_PREFIXES[input.system_prefix()]
                file_path = prefix + input.remote_path()
            else:
                ui.notification_show(f"no valid input choices", type='error')     
                return

            if not Path(file_path).exists():
                ui.notification_show(
                        f"File not found: {file_path}",
                        type="error"
                        )
                return

            ui.notification_show(f"loading from {input.input_type()}")     
            ui.notification_show(f"loading from {file_path}")     

            if input.sample_umis():
                df =(
                        pl.scan_parquet(file_path)
                        .filter(pl.col('reads')>=input.sample_min_reads())
                        .filter(
                            pl.col('umi').is_in(pl.col('umi').sample(input.sample_n_umis()))
                            )	
                        .collect(streaming=True)
                        )

                ui.notification_show(f"finished loading from {file_path}")     

                if df.shape[0] == 0:
                    ui.notification_show(
                            f"No UMIs found with at least {input.sample_min_reads()}",
                            type="error"
                            )
                    return
            else:
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
            df = data()
            umi_reads = df.filter(pl.col('umi') == input.umi()).height
            return f"Selected UMI: {input.umi()}\nNumber of reads: {umi_reads}"
        except Exception as e:
            print(f"Error in selected_umi_stats: {e}")  # Log error
            return "Error calculating UMI stats"

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
    def run_regular_assembly():
        try:
            if data() is None or not input.umi():
                ui.notification_show("Please load data and select a UMI first", type="warning")
                return

            df = data()
            result = (
                    df
                    .pp.assemble_umi(
                        target_umi=input.umi(),
                        k=int(input.kmer_size()),
                        min_cov=int(input.min_coverage()),
                        method=input.assembly_method(),
                        auto_k=input.auto_k(),
                        export_graphs=True,
                        only_largest=True,
                        start_anchor=input.start_anchor(),
                        end_anchor=input.end_anchor(),
                        )
                    )
            print(result)
            handle_assembly_result(result)
        except Exception as e:
            ui.notification_show(f"Error in regular assembly: {str(e)}", type="error")

    @reactive.Effect
    @reactive.event(input.polish)
    def run_polish_assembly():
        try:
            if data() is None or not input.umi():
                ui.notification_show("Please load data and select a UMI first", type="warning")
                return

            base_path = f"{Path(__file__).parent}/{input.umi()}"
            graph_path = f"{base_path}__compressed.csv"
            print(f"will polish contigs from {graph_path}")

            if not Path(graph_path).exists():
                ui.notification_show("Compressed graph data not found", type="error")
                return

            polish_umi = f'polish_{input.umi()}'
            df = (
                    pl.read_csv(graph_path)
                    .with_columns(pl.col('sequence').str.len_chars().alias('length'))
                    .sort('coverage', 'length',  descending=True)
                    .select('node_id', 'sequence', 'coverage', 'length')
                    .with_columns(umi=pl.lit(polish_umi))
                    .rename({'sequence':'r2_seq'})
                    )
            pl.Config().set_tbl_width_chars(100)


            result = df.pp.assemble_umi(target_umi=polish_umi,
                                        k=int(input.kmer_size()),
                                        min_cov=int(input.min_coverage()),
                                        auto_k=input.auto_k(),
                                        export_graphs=True,
                                        only_largest=True,
                                        intbc_5prime='GAGACTGCATGG'
                                        )


            handle_assembly_result(result)
        except Exception as e:
            ui.notification_show(f"Error in polish assembly: {str(e)}", type="error")

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
                        # path_df will be needed for adding decorate to nodes in the graph plot
                        # but assembly_results are to ones proccessed by path_results_output for display
                        print(result)
                        path_results.set({
                            'results':result,
                            'path_nodes': path_df['node_id'].to_list(),
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
    @reactive.event(input.top_path)  
    def path_results_output():
        if path_results() is None:
            return ""
        

        head_n_paths = 5 if input.top_path() == '' else int(input.top_path())

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

    @output
    @render.text 
    @reactive.event(assembly_result)  
    def top_contigs_polish():
        if data() is None or assembly_result() is None:
            return "No polished contigs available"

        try:
            polish_umi = f'polish_{input.umi()}'
            base_path = f"{Path(__file__).parent}/{polish_umi}"
            graph_path = f"{base_path}__compressed.csv"

            if not Path(graph_path).exists():
                return "Polished contig graph data not found"

            df = (
                    pl.read_csv(graph_path)
                    .with_columns(pl.col('sequence').str.len_chars().alias('length'))
                    .sort('coverage', 'length', descending=True)
                    .select('node_id', 'sequence', 'coverage', 'length')
                    .head(5)
                    )

            if not df.shape[0] == 0:
                pl.Config().set_tbl_width_chars(df.get_column('length').max()+1)

            return ui.HTML(format_top_contigs_table(df))

        except Exception as e:
            print(f"Error loading polished top contigs: {str(e)}")
            return "Error loading polished contig data"

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
            filtered_df = (
                    df.filter(pl.col('umi') == input.umi())
                    )

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
    @reactive.event(input.umi, data)

    def coverage_plot():
        mods = pl.read_parquet('mods.parquet')
        ref_str = mods.filter(pl.col('mod')=='mod_0')['seq'][0]

        if data() is None or not input.umi() or input.umi() not in data()['umi']:
            empty_fig = go.Figure(layout=dark_template['layout'])
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


        df =(
                data()
                .filter(pl.col('umi')==input.umi())
                .with_columns(intbc=pl.lit('in')) # dummy intbc field needed for alignment
                )

        result = al.compute_coverage(df , ref_str, max_range=len(ref_str))

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
                **dark_template['layout'],
                xaxis_title="Minimum Coverage",
                yaxis_title="K-mer Size",
                autosize=True,
                )

        # Ensure axis labels are visible
        fig.update_xaxes(showticklabels=True, title_standoff=25)
        fig.update_yaxes(showticklabels=True, title_standoff=25)

        return fig

    @output
    @render_plotly
    def sweep_heatmap():
        if sweep_results() is None:
            # Create empty figure with specific dimensions
            empty_fig = go.Figure(layout=dark_template['layout'])
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
                **dark_template['layout'],
                xaxis_title="Minimum Coverage",
                yaxis_title="K-mer Size",
                coloraxis_colorbar_title="Contig Length"
                )

        # Ensure axis labels are visible
        fig.update_xaxes(showticklabels=True, title_standoff=25)
        fig.update_yaxes(showticklabels=True, title_standoff=25)

        return fig
###
    @output
    @render_plotly
    def assembly_graph():
        if assembly_result() is None:
            empty_fig = go.Figure(layout=dark_template['layout'])
            empty_fig.update_layout(
                    #width=1000,
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

        # Determine graph suffix based on use_polished toggle
        polish_umi = f'polish_{input.umi()}'

        graph_suffix = polish_umi if input.use_polished() else input.umi()

        graph_path = f"{base_path}{graph_suffix}__{input.graph_type()}.dot"

        print(f"Looking for graph at: {graph_path}")  # Debug print

        if not Path(graph_path).exists():
            print(f"Graph file not found: {graph_path}")  # Debug print
            empty_fig = go.Figure(layout=dark_template['layout'])
            empty_fig.update_layout(
                    #width=1000,
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
                line_shape='linear',  # Since we removed line_shape selector
                graph_type=input.graph_type(),
                path_nodes= path_nodes,

                )

        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                    autosize=True,
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

        # Add debug information to layout
        fig.update_layout(
                #title=f"Graph: {graph_path}<br>Polished: {input.use_polished()}",
                annotations=[
                    dict(
                        text=f"Graph: {graph_path}<br>Polished: {input.use_polished()}",
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
