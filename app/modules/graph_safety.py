# modules/graph_safety.py
"""
Safety mechanism for handling large graphs in the FRACTURE Explorer app.
Provides static and simplified rendering options for graphs exceeding size thresholds.
"""

import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
import logging
from typing import Tuple, Dict, Optional, Union
import numpy as np

logger = logging.getLogger("fracture_app.graph_safety")

class GraphSafetyManager:
    """Manages safe rendering of large graphs with configurable thresholds."""
    
    def __init__(self, 
                 node_threshold: int = 1000,
                 edge_threshold: int = 5000,
                 force_static_threshold: int = 5000):
        """
        Initialize safety manager with configurable thresholds.
        
        Args:
            node_threshold: Node count above which to trigger safety mode
            edge_threshold: Edge count above which to trigger safety mode  
            force_static_threshold: Node count above which to force static rendering
        """
        self.node_threshold = node_threshold
        self.edge_threshold = edge_threshold
        self.force_static_threshold = force_static_threshold
        
    def check_graph_size(self, graph: nx.Graph) -> Dict[str, Union[int, bool]]:
        """
        Check graph size and determine rendering strategy.
        
        Returns:
            Dict with size info and recommended rendering mode
        """
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        
        return {
            'node_count': node_count,
            'edge_count': edge_count,
            'requires_safety': node_count > self.node_threshold or edge_count > self.edge_threshold,
            'force_static': node_count > self.force_static_threshold,
            'recommended_mode': self._get_recommended_mode(node_count, edge_count)
        }
    
    def _get_recommended_mode(self, node_count: int, edge_count: int) -> str:
        """Determine recommended rendering mode based on graph size."""
        if node_count > self.force_static_threshold:
            return 'static'
        elif node_count > self.node_threshold:
            return 'simplified'
        else:
            return 'full'
    
    def create_static_rendering(self, graph: nx.Graph, 
                              title: str = "Large Graph (Static View)",
                              sample_size: Optional[int] = None,
                              dark_mode: bool = True,
                              **kwargs) -> go.Figure:
        """
        Create a static, non-interactive rendering for very large graphs with proper node data parsing.
        
        Args:
            graph: NetworkX graph
            title: Title for the plot
            sample_size: If provided, sample this many nodes for visualization
            dark_mode: Whether to use dark theme colors
            **kwargs: Additional visualization arguments
        """
        from .visualization import extract_coverage, parse_node_label, get_node_style
        from .config import SEQUENCE_COLORS
        
        fig = go.Figure()
        
        # Add graph statistics
        stats_text = f"Nodes: {graph.number_of_nodes()}<br>Edges: {graph.number_of_edges()}"
        
        if sample_size and graph.number_of_nodes() > sample_size:
            # Sample nodes for visualization
            sampled_nodes = np.random.choice(list(graph.nodes()), 
                                           size=min(sample_size, graph.number_of_nodes()), 
                                           replace=False)
            subgraph = graph.subgraph(sampled_nodes)
            stats_text += f"<br>Showing {len(sampled_nodes)} sampled nodes"
        else:
            subgraph = graph
            
        # Use faster layout for large graphs
        try:
            if subgraph.number_of_nodes() > 500:
                # Use circular layout for very large graphs (fastest)
                pos = nx.circular_layout(subgraph)
            else:
                # Use spring layout with limited iterations
                pos = nx.spring_layout(subgraph, k=1.0, iterations=20)
        except Exception as e:
            logger.error(f"Layout calculation failed: {e}")
            # Fallback to random layout
            pos = nx.random_layout(subgraph)
        
        # Parse node data similar to create_graph_plot
        node_x, node_y = [], []
        node_colors = []
        node_sizes = []
        coverages = []
        
        # First pass to collect coverage values for scaling
        for node in subgraph.nodes():
            attrs = subgraph.nodes[node]
            label = attrs.get('label', '')
            if isinstance(label, str):
                coverage = extract_coverage(label.strip('"'))
                if coverage is not None:
                    coverages.append(coverage)
        
        # Calculate size scaling factors
        min_coverage = min(coverages) if coverages else 1
        max_coverage = max(coverages) if coverages else 1
        min_size = 4  # Smaller for static view
        max_size = 20  # Smaller for static view
        
        # Second pass to build node properties with proper data parsing
        for node in subgraph.nodes():
            if node not in pos:
                continue
                
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            attrs = subgraph.nodes[node]
            label = attrs.get('label', '')
            
            if isinstance(label, str):
                # Parse node label like in create_graph_plot
                label = label.strip('"').replace('\\\\n', '\n').replace('\\n', '\n')
                id_part, seq_part, cov_part = parse_node_label(label)
                
                # Get node style with proper color based on sequence
                node_style = get_node_style(seq_part, SEQUENCE_COLORS, dark_mode, 
                                          node_id=node, path_nodes=None, 
                                          selected_nodes=None, selected_sequences=None)
                node_colors.append(node_style['color'])
                
                # Calculate node size based on coverage
                coverage = extract_coverage(label)
                if coverage is not None and max_coverage > min_coverage:
                    size = min_size + (max_size - min_size) * (coverage - min_coverage) / (max_coverage - min_coverage)
                else:
                    size = min_size
                node_sizes.append(size)
            else:
                # Fallback for nodes without proper labels
                node_colors.append('rgba(0,0,0,0)')
                node_sizes.append(min_size)
        
        # Create simplified edges (no hover text)
        edge_x = []
        edge_y = []
        for edge in subgraph.edges():
            if edge[0] not in pos or edge[1] not in pos:
                continue
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1.0, color='#8fa1b3' if dark_mode else '#555'),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Add nodes with proper colors and sizes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1.0, color='#c0c5ce' if dark_mode else '#444')
            ),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Update layout with proper theming
        bg_color = '#2d3339' if dark_mode else '#ffffff'
        text_color = '#ffffff' if dark_mode else '#000000'
        
        # Calculate axis ranges based on node positions, excluding outliers
        import numpy as np
        node_x_array = np.array(node_x)
        node_y_array = np.array(node_y)
        
        # Use percentiles to exclude extreme outliers
        x_min, x_max = np.percentile(node_x_array, [1, 99])
        y_min, y_max = np.percentile(node_y_array, [1, 99])
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = 0.05
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(color=text_color)),
            showlegend=False,
            hovermode=False,  # Disable hover for performance
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                range=[x_min - x_range * padding, x_max + x_range * padding]
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                range=[y_min - y_range * padding, y_max + y_range * padding]
            ),
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font_color=text_color,
            annotations=[
                dict(
                    text=stats_text,
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.8)" if not dark_mode else "rgba(0,0,0,0.8)",
                    bordercolor="#888",
                    borderwidth=1,
                    font=dict(color=text_color)
                )
            ]
        )
        
        return fig
    
    def create_simplified_rendering(self, graph: nx.Graph,
                                  max_hover_nodes: int = 100,
                                  **kwargs) -> go.Figure:
        """
        Create a simplified but still interactive rendering.
        
        Args:
            graph: NetworkX graph
            max_hover_nodes: Maximum nodes to include hover text for
            **kwargs: Additional arguments for visualization
        """
        from .visualization import create_graph_plot
        import tempfile
        import os
        
        # Modify kwargs to disable expensive features for simplified rendering
        kwargs['selected_nodes'] = None  # Disable selection for large graphs
        kwargs['selected_sequences'] = None
        kwargs['separate_components'] = False  # Disable component separation
        
        # Create a temporary DOT file from the graph for create_graph_plot
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dot', delete=False) as f:
            nx.drawing.nx_pydot.write_dot(graph, f.name)
            temp_dot_path = f.name
        
        try:
            # Use the existing create_graph_plot function
            fig = create_graph_plot(temp_dot_path, **kwargs)
            return fig
        finally:
            # Clean up temporary file
            if os.path.exists(temp_dot_path):
                os.unlink(temp_dot_path)


def create_safe_graph_widget(dot_path: str, 
                           safety_manager: GraphSafetyManager,
                           user_override: Optional[str] = None,
                           **viz_kwargs) -> Tuple[go.Figure, Dict[str, any]]:
    """
    Create a graph visualization with safety checks.
    
    Args:
        dot_path: Path to DOT file
        safety_manager: GraphSafetyManager instance
        user_override: User-selected rendering mode ('auto', 'full', 'simplified', 'static')
        **viz_kwargs: Additional visualization arguments
        
    Returns:
        Tuple of (figure, metadata dict with rendering info)
    """
    try:
        # Load graph
        graph = nx.drawing.nx_pydot.read_dot(dot_path)
        
        # Check size
        size_info = safety_manager.check_graph_size(graph)
        
        # Determine rendering mode
        if user_override and user_override != 'auto':
            mode = user_override
        else:
            mode = size_info['recommended_mode']
            
        # Log decision
        logger.info(f"Graph size: {size_info['node_count']} nodes, {size_info['edge_count']} edges. "
                   f"Rendering mode: {mode}")
        
        # Create appropriate visualization
        if mode == 'static' or (mode == 'full' and size_info['force_static']):
            fig = safety_manager.create_static_rendering(
                graph, 
                title=f"Large Graph ({size_info['node_count']} nodes)",
                sample_size=1000,
                **viz_kwargs  # Pass through visualization arguments including dark_mode
            )
            actual_mode = 'static'
        elif mode == 'simplified':
            fig = safety_manager.create_simplified_rendering(graph, **viz_kwargs)
            actual_mode = 'simplified'
        else:
            # Full interactive rendering
            from .visualization import create_graph_plot
            fig = create_graph_plot(dot_path, **viz_kwargs)
            actual_mode = 'full'
            
        # Return figure and metadata
        metadata = {
            **size_info,
            'rendering_mode': actual_mode,
            'user_override': user_override
        }
        
        return fig, metadata
        
    except Exception as e:
        logger.error(f"Error creating safe graph: {e}")
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading graph: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig, {'error': str(e)}
