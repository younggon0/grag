"""
Graph Visualizer Module
Creates interactive graph visualizations using Pyvis
"""

import networkx as nx
from pyvis.network import Network
import tempfile
import os
from typing import Dict, Any

class GraphVisualizer:
    def __init__(self):
        """Initialize graph visualizer"""
        self.color_map = {
            'PERSON': '#FF6B6B',      # Red
            'ORGANIZATION': '#4ECDC4', # Teal
            'CONCEPT': '#45B7D1',      # Blue
            'LOCATION': '#95E77E',     # Green
            'EVENT': '#FFA07A',        # Light Salmon
            'PRODUCT': '#DDA0DD',      # Plum
            'UNKNOWN': '#C0C0C0'       # Silver
        }
    
    def create_pyvis_graph(
        self,
        graph: nx.DiGraph,
        physics: bool = True,
        show_labels: bool = True,
        height: str = "600px",
        width: str = "100%"
    ) -> str:
        """
        Create an interactive Pyvis graph
        
        Args:
            graph: NetworkX graph
            physics: Enable physics simulation
            show_labels: Show node labels
            height: Graph height
            width: Graph width
            
        Returns:
            HTML string of the graph
        """
        if graph.number_of_nodes() == 0:
            return "<p>No graph data to visualize</p>"
        
        # Create Pyvis network
        net = Network(
            height=height,
            width=width,
            directed=True,
            notebook=False,
            bgcolor="#ffffff",
            font_color="#000000"
        )
        
        # Configure physics
        if physics:
            net.barnes_hut(
                gravity=-80000,
                central_gravity=0.3,
                spring_length=100,
                spring_strength=0.001,
                damping=0.09
            )
        else:
            net.toggle_physics(False)
        
        # Limit nodes for performance
        if graph.number_of_nodes() > 500:
            # Get top nodes by degree
            degrees = dict(graph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:500]
            top_node_names = [name for name, _ in top_nodes]
            graph = graph.subgraph(top_node_names).copy()
        
        # Add nodes
        for node, attrs in graph.nodes(data=True):
            node_type = attrs.get('type', 'UNKNOWN')
            color = self.color_map.get(node_type, self.color_map['UNKNOWN'])
            
            # Calculate node size based on degree
            degree = graph.degree(node)
            size = min(10 + degree * 3, 50)  # Size between 10 and 50
            
            # Prepare label
            label = node if show_labels else ""
            
            # Prepare hover text
            description = attrs.get('description', 'No description')
            title = f"<b>{node}</b><br>Type: {node_type}<br>Connections: {degree}<br>{description[:100]}..."
            
            net.add_node(
                node,
                label=label,
                color=color,
                size=size,
                title=title,
                font={'size': 12}
            )
        
        # Add edges
        for source, target, attrs in graph.edges(data=True):
            edge_type = attrs.get('type', 'relates_to')
            edge_label = edge_type if show_labels else ""
            
            # Prepare hover text for edge
            description = attrs.get('description', '')
            title = f"{edge_type}: {description[:100]}..." if description else edge_type
            
            net.add_edge(
                source,
                target,
                label=edge_label,
                title=title,
                arrows='to',
                color={'color': '#888888', 'opacity': 0.6},
                width=1
            )
        
        # Set options
        net.set_options("""
        var options = {
            "nodes": {
                "borderWidth": 2,
                "borderWidthSelected": 3,
                "shape": "dot",
                "font": {
                    "size": 12,
                    "strokeWidth": 2,
                    "strokeColor": "#ffffff"
                }
            },
            "edges": {
                "smooth": {
                    "type": "continuous",
                    "forceDirection": "none"
                },
                "font": {
                    "size": 10,
                    "align": "middle"
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 200,
                "navigationButtons": true,
                "keyboard": true
            },
            "manipulation": {
                "enabled": false
            }
        }
        """)
        
        # Generate HTML
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as tmp:
            net.save_graph(tmp.name)
            tmp_path = tmp.name
        
        # Read the HTML
        with open(tmp_path, 'r') as f:
            html_content = f.read()
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return html_content
    
    def create_matplotlib_graph(self, graph: nx.DiGraph) -> Any:
        """
        Create a simple matplotlib graph as fallback
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        if graph.number_of_nodes() == 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No graph data to visualize', 
                   ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        # Limit nodes for visualization
        if graph.number_of_nodes() > 100:
            degrees = dict(graph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:100]
            top_node_names = [name for name, _ in top_nodes]
            graph = graph.subgraph(top_node_names).copy()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate layout
        pos = nx.spring_layout(graph, k=2, iterations=50)
        
        # Draw nodes
        node_colors = []
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'UNKNOWN')
            color = self.color_map.get(node_type, self.color_map['UNKNOWN'])
            node_colors.append(color)
        
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_colors,
            node_size=300,
            alpha=0.9,
            ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            graph, pos,
            edge_color='gray',
            arrows=True,
            alpha=0.5,
            ax=ax,
            arrowsize=20
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            graph, pos,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
        
        # Add legend
        legend_elements = []
        for entity_type, color in self.color_map.items():
            if entity_type != 'UNKNOWN':
                from matplotlib.patches import Patch
                legend_elements.append(Patch(facecolor=color, label=entity_type))
        
        ax.legend(handles=legend_elements, loc='upper left')
        ax.set_title("Knowledge Graph Visualization", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig