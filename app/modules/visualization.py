# Visualization functions
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import re
import html
import numpy as np
import polars as pl
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

def get_node_style(seq, sequence_colors, dark_mode, node_id=None, path_nodes=None, selected_nodes=None, selected_sequences=None):
    """
    Determine node style including color and line properties based on node properties.
    
    Args:
        seq: The sequence to check for anchor sequences
        sequence_colors: Dict mapping anchor sequences to their colors
        dark_mode: Boolean indicating if dark mode is enabled
        node_id: The ID of the node (used for path checking)
        path_nodes: Set/list of node IDs that are part of the path
        selected_nodes: Set/list of node IDs that should be highlighted
        selected_sequences: Set/list of sequences that should be highlighted
    
    Returns:
        dict: Style properties for the node
    """
    # HIGHEST PRIORITY: Check if node is selected by ID or sequence
    is_selected = False
    
    # Check if node is in selected nodes list
    if selected_nodes is not None and node_id is not None and node_id in selected_nodes:
        is_selected = True
    
    # Check if sequence is in selected sequences list
    if selected_sequences is not None and seq is not None:
        for selected_seq in selected_sequences:
            if selected_seq.strip() and selected_seq.strip().upper() in seq.upper():
                is_selected = True
                break
    
    if is_selected:
        return dict(
            color='rgba(255, 0, 0, 0.8)',  # Bright red with transparency
            line=dict(color='#ff0000', width=4)  # Thick red border
        )
    
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
        try:
            # Ensure both nodes have positions
            if u not in pos or v not in pos:
                print(f"Warning: Edge nodes ({u}, {v}) missing from position dictionary, skipping")
                continue
                
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
        except Exception as e:
            print(f"Error creating edge for ({u}, {v}): {str(e)}")
            continue
            
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
        clickmode='event+select',  # Enable click events
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

def create_graph_plot(dot_path, dark_mode=True, line_shape='linear', graph_type='compressed',
                      debug=False, path_nodes=None, weighted=False, weight_method='nlog',
                      separate_components=False, component_padding=3.0, min_component_size=3, spring_args=None,
                      selected_nodes=None, selected_sequences=None, precalculated_positions=None, layout_algorithm="fruchterman_reingold",
                      start_anchor=None, end_anchor=None):
    """Create an interactive plot of the assembly graph.

    Args:
        dot_path (str): Path to the DOT file
        dark_mode (bool): Whether to use dark theme colors
        line_shape (str): Shape of edges ('linear' or 'spline')
        graph_type (str): Type of graph visualization
        debug (bool): Whether to print debug information
        path_nodes (list/set): Node IDs that are part of the path
        separate_components (bool): Whether to position disjoint graphs separately
        component_padding (float): Amount of padding between separate components
        min_component_size (int): Minimum size of a component to be included in the plot
        selected_nodes (list/set): Node IDs that should be highlighted with selection styling
        selected_sequences (list/set): Sequences that should be highlighted with selection styling
        start_anchor (str): Start anchor sequence to highlight (optional)
        end_anchor (str): End anchor sequence to highlight (optional)
    """
    if spring_args is None:
        spring_args = {'k': 1.5, 'iterations': 50, 'scale': 2.0}

    # Build sequence colors dictionary dynamically from anchor parameters
    # Fall back to SEQUENCE_COLORS if anchors not provided
    sequence_colors = {}
    if start_anchor:
        sequence_colors[start_anchor] = '#50C878'  # Emerald green
    if end_anchor:
        sequence_colors[end_anchor] = '#9370DB'  # Medium purple

    # If no anchors provided, use default hardcoded values
    if not sequence_colors:
        sequence_colors = SEQUENCE_COLORS
    
    try:
        graph = nx.drawing.nx_pydot.read_dot(dot_path)
        if len(graph.nodes()) == 0:
            return create_empty_figure("Empty graph - no nodes found")
    except Exception as e:
        return create_empty_figure(f"Error creating graph: {str(e)}")

    if weighted:
        graph = create_weighted_graph(graph, weight_method)
    
    
    # Calculate node positions based on layout settings
    if precalculated_positions is not None:
        pos = precalculated_positions
    else:
        if separate_components:
            # Identify connected components for separate layout
            components = list(nx.connected_components(graph.to_undirected()))
            
            # Filter out components smaller than min_component_size
            filtered_components = [comp for comp in components if len(comp) >= min_component_size]
            print(f"Graph has {len(components)} connected components, {len(filtered_components)} meet the minimum size requirement")
            
            # Sort components by size (largest first)
            filtered_components.sort(key=len, reverse=True)
            
            # Calculate layout for each component and place in a grid
            pos = {}
            component_info = []  # Store component info for grid placement
            
            # First pass - calculate layouts and store info
            for component in filtered_components:
                # Create subgraph for this component
                subgraph = graph.subgraph(component)
                
                # Calculate layout for this component using selected algorithm
                component_pos = get_layout_algorithm(layout_algorithm, subgraph, spring_args, weighted=weighted)
                
                # Find bounding box and scale factor based on component size
                min_x = min(p[0] for p in component_pos.values()) if component_pos else 0
                max_x = max(p[0] for p in component_pos.values()) if component_pos else 0
                min_y = min(p[1] for p in component_pos.values()) if component_pos else 0
                max_y = max(p[1] for p in component_pos.values()) if component_pos else 0
                width = max_x - min_x
                height = max_y - min_y
                
                # Normalize the scale based on component size
                # Calculate a scale factor proportional to sqrt(component size)
                # This ensures larger components get more space but not excessively so
                scale_factor = np.sqrt(len(component)) / 2.0
                
                component_info.append({
                    'component': component,
                    'pos': component_pos,
                    'width': width * scale_factor,
                    'height': height * scale_factor,
                    'scale_factor': scale_factor,
                    'size': len(component)
                })
            
            # If no components meet the size requirement, use selected algorithm for full graph
            if not filtered_components:
                pos = get_layout_algorithm(layout_algorithm, graph, spring_args, weighted=weighted)
            else:
                # Second pass - arrange components in a grid, starting with largest component at top left
                current_x, current_y = 0, 0
                
                # Group components into rows
                rows = []
                current_row = []
                current_row_width = 0
                max_row_width = 20.0  # Maximum width for a row before starting a new row
                
                for info in component_info:
                    # If adding this component would exceed the max row width, start a new row
                    if current_row_width + info['width'] + component_padding > max_row_width and current_row:
                        rows.append(current_row)
                        current_row = [info]
                        current_row_width = info['width']
                    else:
                        current_row.append(info)
                        current_row_width += info['width'] + component_padding
                
                # Add the last row if not empty
                if current_row:
                    rows.append(current_row)
                
                # Position components within rows
                current_y = 0
                for row in rows:
                    current_x = 0
                    max_height_in_row = max(info['height'] for info in row) if row else 0
                    
                    for info in row:
                        component = info['component']
                        component_pos = info['pos']
                        scale_factor = info['scale_factor']
                        
                        # Apply scaling and offset to this component's positions
                        for node, (x, y) in component_pos.items():
                            # Scale the position
                            scaled_x = x * scale_factor
                            scaled_y = y * scale_factor
                            
                            # Apply offset
                            pos[node] = (scaled_x + current_x, scaled_y + current_y)
                        
                        # Move to the next position in the row
                        current_x += info['width'] + component_padding
                    
                    # Move to the next row
                    current_y += max_height_in_row + component_padding
        else:
            # Use selected algorithm for the entire graph
            pos = get_layout_algorithm(layout_algorithm, graph, spring_args, weighted=weighted)
    
    # Extract node information
    node_x, node_y = [], []
    node_labels = []
    node_colors = []
    node_sizes = []
    hover_texts = []
    custom_data = []  # Store node IDs for click handling
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
        try:
            # Check if node is in the position dictionary
            if node not in pos:
                print(f"Warning: Node {node} has no position, skipping")
                continue
                
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            custom_data.append(node)  # Store node ID for click handling

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
                                          node_id=node, path_nodes=path_nodes, 
                                          selected_nodes=selected_nodes, selected_sequences=selected_sequences)
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
        except Exception as e:
            print(f"Error processing node {node}: {str(e)}")
            # Don't return the node ID here - continue processing other nodes
            continue

    # Create edges
    edge_x, edge_y, edge_texts = create_edges(graph, pos, line_shape)

    # Create the figure
    try:
        # Check if we have any nodes to display
        if not node_x or not node_y:
            print("Warning: No nodes with valid positions found")
            fig = create_empty_figure("No nodes with valid positions found")
            return fig
            
        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='#4f5b66' if dark_mode else '#888', shape=line_shape),
            hoverinfo='text',
            hovertext=edge_texts,
            mode='lines',
            showlegend=False,
            name='edges'
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
            mode='markers+text',
            hoverinfo='text',
            hovertext=hover_texts,
            text=node_labels,
            textposition="bottom center",
            textfont=dict(size=12),
            marker=node_marker,
            customdata=custom_data,  # This MUST be the node IDs
            showlegend=False,
            name='nodes',
            hoveron='points',
            # Add this to ensure selection works:
            selectedpoints=[],
            selected=dict(marker=dict(color='red', size=15)),
            unselected=dict(marker=dict(opacity=0.7))
        ))

        # Update layout
        update_figure_layout(fig, dark_mode, node_x, node_y)
        
        # Configure for Shiny integration - enable events
        fig.update_layout(
            uirevision=True,  # Preserve UI state
            clickmode='event+select',  # Ensure click events are enabled
            dragmode='pan',  # Set drag mode to pan to avoid conflicts with selection
        )
        
        # Ensure all traces can generate click events
        for trace in fig.data:
            if hasattr(trace, 'name') and trace.name == 'nodes':
                trace.update(
                    selected=dict(marker=dict(color='red', size=15)),  # Configure selection appearance
                    unselected=dict(marker=dict(opacity=0.7))  # Configure unselected appearance
                )
        
        # Make absolutely sure we're returning a proper Plotly figure
        if not isinstance(fig, go.Figure):
            print(f"Error: Expected go.Figure but got {type(fig)}")
            return create_empty_figure("Error creating graph visualization")
            
        return fig  # Return Figure object, conversion to FigureWidget happens in render function
    except Exception as e:
        print(f"Error creating figure: {str(e)}")
        return create_empty_figure(f"Error creating graph: {str(e)}")

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
