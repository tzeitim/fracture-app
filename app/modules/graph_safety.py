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
                              sample_size: Optional[int] = None) -> go.Figure:
        """
        Create a static, non-interactive rendering for very large graphs.
        
        Args:
            graph: NetworkX graph
            title: Title for the plot
            sample_size: If provided, sample this many nodes for visualization
        """
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
        
        # Extract positions
        node_x = [pos[node][0] for node in subgraph.nodes()]
        node_y = [pos[node][1] for node in subgraph.nodes()]
        
        # Create simplified edges (no hover text)
        edge_x = []
        edge_y = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Add nodes (no labels or hover for performance)
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=3,
                color='#4895fa',
                line=dict(width=0)
            ),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5),
            showlegend=False,
            hovermode=False,  # Disable hover for performance
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            annotations=[
                dict(
                    text=stats_text,
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#888",
                    borderwidth=1
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
                sample_size=1000
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
