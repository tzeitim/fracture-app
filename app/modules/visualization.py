# Visualization functions
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import re
import html
import numpy as np
from .config import DARK_TEMPLATE, SEQUENCE_COLORS

def extract_coverage(input_data):
    """Extract coverage value from either a node label string or graph."""
    if isinstance(input_data, str):
        # Handle string input (node label)
        match = re.search(r'cov:\s*(\d+)', input_data)
        return int(match.group(1)) if match else None
    else:
        # Handle graph input
        coverages = {}
        for node in input_data.nodes():
            attrs = input_data.nodes[node]
            label = attrs.get('label', '')
            if isinstance(label, str):
                match = re.search(r'cov:\s*(\d+)', label.strip('"'))
                if match:
                    coverages[node] = int(match.group(1))
        return coverages

def extract_seqs(graph):
    coverages = {}    # First pass to collect coverage values

    for node in graph.nodes():
        attrs = graph.nodes[node]
        label = attrs.get('label', '')
        if isinstance(label, str):
            coverage = extract_attr_from_label(label.strip('"'), r'Seq:\s*(.+)\\n')
            if coverage is not None:
                coverages[node] = coverage
    return coverages

def extract_attr_from_label(label, attr_re):
    """Extract coverage value from node label."""
    if not label or not isinstance(label, str):
        return None
    import re
    match = re.search(attr_re, label)
    if match:
        return match.group(1)
    return None

def create_weighted_graph(graph, weight_method):
    nodes = graph.nodes
    edges = graph.edges
    coverages = extract_coverage(graph)
    seqs = extract_seqs(graph)

    G = nx.DiGraph()
    
    # First add all nodes with their attributes
    for node, attrs in graph.nodes(data=True):
        G.add_node(node, **attrs)  # This preserves all node attributes including labels

    # Then add weighted edges
    for u, v, edge_attrs in edges(data=True):
        if weight_method == "nlog":
            avg_coverage = (int(coverages[u]) + int(coverages[v])) / 2
            weight = -np.log(avg_coverage)
        if weight_method == "inverse":
            weight = 2.0 / (int(coverages[u]) + int(coverages[v])) 
        
        # Add edge with both weight and original attributes
        edge_data = edge_attrs.copy()
        edge_data['weight'] = weight
        G.add_edge(u, v, **edge_data)

    return G

def get_node_style(seq, sequence_colors, dark_mode, node_id=None, path_nodes=None):
    """
    Determine node style including color and line properties based on node properties.
    
    Args:
        seq: The sequence to check for anchor sequences
        sequence_colors: Dict mapping anchor sequences to their colors
        dark_mode: Boolean indicating if dark mode is enabled
        node_id: The ID of the node (used for path checking)
        path_nodes: Set/list of node IDs that are part of the path
    
    Returns:
        dict: Style properties for the node
    """
    # Default style for empty/invalid sequences
    if not seq:
        return dict(
            color='rgba(0,0,0,0)', 
            line=dict(color='#4f5b66' if dark_mode else '#888', width=1)
        )

    # Clean up sequence string
    seq = seq.strip('"').strip()
    
    # Priority 1: Check if node is in path but doesn't contain anchor sequences
    if path_nodes is not None and node_id is not None and node_id in path_nodes:
        if not any(target_seq in seq for target_seq in sequence_colors.keys()):
            return dict(
                color='rgba(255, 215, 0, 0.15)', # yellow 30 alpha
                line=dict(color='#4f5b66' if dark_mode else '#888', width=1)
            )
    
    # Priority 2: Check for presence of both anchor sequences
    has_both = all(target_seq in seq for target_seq in sequence_colors.keys())
    if has_both:
        return dict(
            color='#FFA500',  # Orange for nodes with both anchors
            line=dict(color='#4f5b66' if dark_mode else '#888', width=1)
        )
    
    # Priority 3: Check for individual anchor sequences
    for target_seq, color in sequence_colors.items():
        if target_seq in seq:
            return dict(
                color=color,
                line=dict(color='#4f5b66' if dark_mode else '#888', width=1)
            )
    
    # Default style for nodes without special properties
    return dict(
        color='rgba(0,0,0,0)',
        line=dict(color='#4f5b66' if dark_mode else '#888', width=1)
    )

def create_empty_figure(message):
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        height=1000,
        autosize=True,
        annotations=[dict(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color='#ffffff', size=14)
        )]
    )
    return fig

def parse_node_label(label):
    """Parse node label into components."""
    id_part = ''
    seq_part = ''
    cov_part = ''
    
    for part in label.split('\n'):
        if 'ID:' in part:
            id_part = part.replace('ID:', '').strip()
        elif 'Seq:' in part:
            seq_part = part.replace('Seq:', '').strip()
        elif 'cov:' in part:
            cov_part = part.replace('cov:', '').strip()
            
    return id_part, seq_part, cov_part

def format_sequence_with_highlights(seq_part, sequence_colors, line_length=60):
    """Format sequence with highlights, properly handling HTML tags.
    
    Args:
        seq_part (str): Sequence to format
        sequence_colors (dict): Mapping of sequences to their highlight colors
        line_length (int): Maximum length for each line
        
    Returns:
        str: Formatted HTML string with highlights
    """
    import html
    
    # First escape any HTML in the original text
    escaped_text = html.escape(seq_part)
    
    # Find all sequences to highlight and their positions
    replacements = []
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
    
    # Sort replacements in reverse order to avoid position shifts
    replacements.sort(key=lambda x: x[0], reverse=True)
    
    # Apply replacements
    result = escaped_text
    for start, end, html_span in replacements:
        result = result[:start] + html_span + result[end:]
    
    # Split into lines with proper HTML tag handling
    lines = []
    current_line = []
    char_count = 0
    in_tag = False
    tag_buffer = []
    
    for char in result:
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
        
    if tag_buffer:  # Handle any unclosed tags
        lines[-1] += ''.join(tag_buffer)
        
    return '<br>'.join(lines)

def build_hover_text(id_part, seq_part, cov_part, sequence_colors):
    """Build hover text with proper formatting."""
    hover_text = []
    if id_part:
        hover_text.append(f"<b>ID:</b> {html.escape(id_part)}")
    if seq_part:
        highlighted_seq = format_sequence_with_highlights(seq_part, sequence_colors)
        hover_text.append(f"<b>Sequence:</b><br>{highlighted_seq}")
    if cov_part:
        hover_text.append(f"<b>Coverage:</b> {html.escape(cov_part)}")
    return '<br>'.join(hover_text)

def create_edges(graph, pos, line_shape):
    """Create edge traces for the graph."""
    edge_x, edge_y, edge_texts = [], [], []
    
    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        if line_shape == 'linear':
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        else:
            # Add curved edges
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            offset = 0.05
            if abs(x1 - x0) > abs(y1 - y0):
                mid_y += offset
            else:
                mid_x += offset
            edge_x.extend([x0, mid_x, x1, None])
            edge_y.extend([y0, mid_y, y1, None])
            
        # Add edge label
        try:
            edge_data = graph.get_edge_data(u, v)
            label = edge_data.get('label', '') if edge_data else ''
            if isinstance(label, str):
                label = label.strip('"')
            edge_texts.append(f"<b>{u} → {v}</b><br>{label}" if label else f"<b>{u} → {v}</b>")
        except Exception as e:
            edge_texts.append(f"<b>{u} → {v}</b>")
            
    return edge_x, edge_y, edge_texts

def update_figure_layout(fig, dark_mode, node_x, node_y):
    """Update the figure layout with proper styling."""
    bg_color = '#2d3339' if dark_mode else '#ffffff'
    text_color = '#ffffff' if dark_mode else '#000000'
    
    # Calculate padding and ranges
    x_range = max(node_x) - min(node_x)
    y_range = max(node_y) - min(node_y)
    padding = 0.02
    
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
        margin=dict(b=40, l=20, r=20, t=40),
        annotations=[dict(
            text="Assembly Graph",
            showarrow=False,
            xref="paper", 
            yref="paper",
            x=0.5, 
            y=1.02,
            font=dict(size=16, color=text_color)
        )],
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[min(node_x) - x_range * padding, max(node_x) + x_range * padding]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[min(node_y) - y_range * padding, max(node_y) + y_range * padding]
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

def create_graph_plot(dot_path, dark_mode=True, line_shape='linear', graph_type='compressed', 
                      debug=False, path_nodes=None, weighted=False, weight_method='nlog', 
                      spring_args=None):
    """Create an interactive plot of the assembly graph.
    
    Args:
        dot_path (str): Path to the DOT file
        dark_mode (bool): Whether to use dark theme colors
        line_shape (str): Shape of edges ('linear' or 'spline')
        graph_type (str): Type of graph visualization
        debug (bool): Whether to print debug information
        path_nodes (list/set): Node IDs that are part of the path
    """
    if spring_args is None:
        spring_args = {'k': 1.5, 'iterations': 50, 'scale': 2.0} 

    # Define sequences and their colors
    sequence_colors = SEQUENCE_COLORS
    
    try:
        graph = nx.drawing.nx_pydot.read_dot(dot_path)
        if len(graph.nodes()) == 0:
            return create_empty_figure("Empty graph - no nodes found")
    except Exception as e:
        return create_empty_figure(f"Error creating graph: {str(e)}")

    if weighted:
        graph = create_weighted_graph(graph, weight_method)
        pos = nx.spring_layout(graph, weight='weight', **spring_args, seed=42)
    else:
        # Calculate layout
        pos = nx.kamada_kawai_layout(graph.to_undirected(), scale=2.0)
    
    # Extract node information
    node_x, node_y = [], []
    node_labels = []
    node_colors = []
    node_sizes = []
    hover_texts = []
    coverages = []

    # First pass to collect coverage values for scaling
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

    # Second pass to build node properties
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        attrs = graph.nodes[node]
        label = attrs.get('label', '')
        
        if isinstance(label, str):
            # Parse node label
            label = label.strip('"').replace('\\\\n', '\n').replace('\\n', '\n')
            id_part, seq_part, cov_part = parse_node_label(label)
            
            # Build hover text
            hover_text = build_hover_text(id_part, seq_part, cov_part, sequence_colors)
            hover_texts.append(hover_text)
            
            # Get node style
            node_style = get_node_style(seq_part, sequence_colors, dark_mode, 
                                      node_id=seq_part, path_nodes=path_nodes)
            node_colors.append(node_style['color'])
            
            # Calculate node size
            coverage = extract_coverage(label)
            if coverage is not None and max_coverage > min_coverage:
                size = min_size + (max_size - min_size) * (coverage - min_coverage) / (max_coverage - min_coverage)
            else:
                size = min_size
            node_sizes.append(size)
            
            node_labels.append("")  # Empty label since we're using hover text
        else:
            hover_texts.append(f"Node: {node}")
            node_colors.append('rgba(0,0,0,0)')
            node_sizes.append(min_size)
            node_labels.append(str(node))

    # Create edges
    edge_x, edge_y, edge_texts = create_edges(graph, pos, line_shape)

    # Create figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#4f5b66' if dark_mode else '#888', shape=line_shape),
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
        line=dict(width=1, color='#4f5b66' if dark_mode else '#888'),
        symbol='circle'
    )

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=hover_texts,
        text=node_labels,
        textposition="bottom center",
        textfont=dict(size=12),
        marker=node_marker,
        showlegend=False
    ))

    # Update layout
    update_figure_layout(fig, dark_mode, node_x, node_y)
    
    return fig

def highlight_sequences_in_table(text):
    """Highlight specific sequences in text with HTML formatting."""
    sequence_colors = SEQUENCE_COLORS

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

def create_coverage_distribution_plot(df):
    """Create a coverage distribution plot"""
    if df is None:
        return go.Figure()

    df_plot = (df
              .select('umi', 'reads')
              .unique())

    fig = px.ecdf(
        df_plot.to_pandas(),
        y='reads',
        ecdfmode="reversed",
    )

    fig.update_layout(
        **DARK_TEMPLATE['layout'],
        xaxis_type="log",
        yaxis_type="log", 
        xaxis_title="Fraction of UMIs",
        yaxis_title="Number of Reads",
        title="Reads per UMI Distribution (Complementary CDF)"
    )

    return fig

def create_reads_per_umi_plot(df):
    """Create a bar chart of top UMIs by read count"""
    if df is None:
        return go.Figure()

    reads_per_umi = (
            df
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
            **DARK_TEMPLATE['layout'],
            xaxis_tickangle=45,
            showlegend=False,
            xaxis_title="UMI",
            yaxis_title="Read Count"
            )

    # Update bars color to match theme
    fig.update_traces(marker_color='#375a7f')  # Blue shade matching Slate theme

    return fig
