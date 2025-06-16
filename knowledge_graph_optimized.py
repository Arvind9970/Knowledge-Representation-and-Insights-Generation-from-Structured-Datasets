"""
Optimized Knowledge Graph implementation with improved visualization.
"""
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import time
from pathlib import Path

try:
    from pyvis.network import Network
    PYWIS_AVAILABLE = True
except ImportError:
    PYWIS_AVAILABLE = False

class KnowledgeGraph:
    """Optimized Knowledge Graph with improved visualization capabilities."""
    
    def __init__(self):
        """Initialize the knowledge graph."""
        self.graph = nx.Graph()
        self.entities = {}
        self.relationships = {}
        self.logger = logging.getLogger(__name__)
        self._layout_cache = None  # Cache for computed layouts

    def add_entity(self, entity_id: str, entity_type: str, attributes: Dict) -> None:
        """Add an entity to the graph.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of the entity (e.g., 'vehicle', 'manufacturer')
            attributes: Dictionary of attributes for the entity
        """
        # Remove 'type' from attributes if it exists to avoid conflict
        attrs = {k: v for k, v in attributes.items() if k != 'type'}
        # Add the entity with type and other attributes
        self.graph.add_node(entity_id, node_type=entity_type, **attrs)
        self.entities[entity_id] = {'type': entity_type, 'attributes': attrs}

    def add_relationship(self, source: str, target: str, 
                        relationship_type: str, 
                        attributes: Dict[str, Any] = None) -> None:
        """Add a relationship between entities."""
        if attributes is None:
            attributes = {}
        self.graph.add_edge(source, target, type=relationship_type, **attributes)
        self.relationships[(source, target)] = {'type': relationship_type, 'attributes': attributes}

    def _get_node_color(self, node_data) -> str:
        """Get color for a node based on its type.
        
        Args:
            node_data: Either a node data dictionary or a node type string
            
        Returns:
            Hex color code for the node type
        """
        color_map = {
            'vehicle': '#4285F4',       # Blue
            'manufacturer': '#EA4335',  # Red
            'state': '#34A853',         # Green
            'type': '#FBBC05',         # Yellow
            'other': '#9E9E9E'         # Gray
        }
        
        # Handle both dictionary and string inputs
        if isinstance(node_data, dict):
            # Get the node type from dictionary
            node_type = node_data.get('node_type', 'other')
        else:
            # Assume it's already a node type string
            node_type = str(node_data)
            
        # Get the base type (first part before any underscores)
        base_type = node_type.split('_')[0].lower()
        return color_map.get(base_type, color_map['other'])

    def _sample_graph(self, max_nodes: int = 100, max_edges: int = 200) -> nx.Graph:
        """Sample a subgraph if the graph is too large."""
        if len(self.graph.nodes) <= max_nodes and len(self.graph.edges) <= max_edges:
            return self.graph.copy()
            
        self.logger.warning(
            f"Graph is too large ({len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges). "
            f"Sampling to {max_nodes} nodes and {max_edges} edges."
        )
        
        # Get the most connected nodes (hubs)
        degrees = dict(self.graph.degree())
        top_nodes = [n for n, _ in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes//2]]
        
        # Add some random nodes for diversity
        remaining_nodes = list(set(self.graph.nodes()) - set(top_nodes))
        if remaining_nodes:
            random_nodes = list(np.random.choice(
                remaining_nodes,
                size=min(len(remaining_nodes), max_nodes - len(top_nodes)),
                replace=False
            ))
            selected_nodes = set(top_nodes + random_nodes)
        else:
            selected_nodes = set(top_nodes)
        
        # Create subgraph with selected nodes and their connections
        subgraph = self.graph.subgraph(selected_nodes).copy()
        
        # If still too many edges, sample them
        if len(subgraph.edges) > max_edges:
            edges = list(subgraph.edges(data=True))
            selected_edges = edges[:max_edges]
            sampled_graph = nx.Graph()
            sampled_graph.add_nodes_from((n, subgraph.nodes[n]) for n in subgraph.nodes())
            sampled_graph.add_edges_from((u, v, d) for u, v, d in selected_edges)
            return sampled_graph
            
        return subgraph

    def _get_layout(self, graph: nx.Graph, layout: str = 'kamada_kawai') -> Dict:
        """Compute and cache graph layout."""
        cache_key = f"{layout}_{len(graph.nodes)}_{len(graph.edges)}"
        
        if self._layout_cache and self._layout_cache.get('key') == cache_key:
            return self._layout_cache['pos']
            
        self.logger.info(f"Computing {layout} layout for {len(graph.nodes)} nodes...")
        start_time = time.time()
        
        try:
            if layout == 'spring':
                pos = nx.spring_layout(graph, k=1.5/np.sqrt(len(graph.nodes)), iterations=50)
            elif layout == 'circular':
                pos = nx.circular_layout(graph)
            elif layout == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(graph)
            elif layout == 'fruchterman_reingold':
                pos = nx.fruchterman_reingold_layout(graph, k=0.1)
            else:
                pos = nx.random_layout(graph)
                
            self._layout_cache = {'key': cache_key, 'pos': pos}
            self.logger.info(f"Layout computed in {time.time() - start_time:.2f} seconds")
            return pos
            
        except Exception as e:
            self.logger.error(f"Error computing layout: {e}")
            return nx.random_layout(graph)

    def _compute_layout(self, graph: nx.Graph, layout: str = 'spring') -> Dict:
        """Compute graph layout."""
        if layout == 'spring':
            pos = nx.spring_layout(graph, k=1.5/np.sqrt(len(graph.nodes)), iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)
        elif layout == 'fruchterman_reingold':
            pos = nx.fruchterman_reingold_layout(graph, k=0.1)
        else:
            pos = nx.random_layout(graph)
        return pos

    def _get_edge_color(self, edge_type: str) -> str:
        """Get color for an edge based on its type."""
        color_map = {
            'related': '#9E9E9E',  # Gray
            'manufactured_by': '#EA4335',  # Red
            'registered_in': '#34A853',  # Green
            'other': '#4285F4'  # Blue
        }
        return color_map.get(edge_type, color_map['other'])

    def visualize_graph(self, 
                      layout: str = 'spring', 
                      max_nodes: int = 100, 
                      max_edges: int = 200,
                      output_file: str = None) -> None:
        """Generate a simple graph visualization.
        
        Args:
            layout: Layout algorithm to use (spring, circular, kamada_kawai, random, fruchterman_reingold)
            max_nodes: Maximum number of nodes to display
            max_edges: Maximum number of edges to display
            output_file: Path to save the visualization (if None, displays the plot)
        """
        try:
            # Sample the graph if it's too large
            if len(self.graph.nodes) > max_nodes or len(self.graph.edges) > max_edges:
                graph = self._sample_graph(max_nodes, max_edges)
            else:
                graph = self.graph
            
            # Get layout
            pos = self._compute_layout(graph, layout)
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Draw edges
            nx.draw_networkx_edges(
                graph, 
                pos, 
                edge_color='gray',
                alpha=0.3,
                width=0.5
            )
            
            # Draw nodes with basic coloring
            node_colors = [self._get_node_color(data) for _, data in graph.nodes(data=True)]
            nx.draw_networkx_nodes(
                graph, 
                pos, 
                node_size=50,
                node_color=node_colors,
                alpha=0.8
            )
            
            # Add basic labels for a few nodes (not all to avoid clutter)
            labels = {}
            for node, data in list(graph.nodes(data=True))[:20]:  # Label first 20 nodes only
                if 'model' in data and 'manufacturer' in data:
                    labels[node] = f"{data['manufacturer']} {data['model']}"
                elif 'model' in data:
                    labels[node] = data['model']
                elif 'manufacturer' in data:
                    labels[node] = data['manufacturer']
                else:
                    labels[node] = str(node)
            
            nx.draw_networkx_labels(
                graph, 
                pos, 
                labels,
                font_size=8,
                font_color='black'
            )
            
            plt.title(f'Vehicle Knowledge Graph\n{len(graph.nodes)} nodes, {len(graph.edges)} edges')
            plt.axis('off')
            
            # Add simple legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4285F4', markersize=10, label='Vehicles'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#EA4335', markersize=10, label='Manufacturers'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#34A853', markersize=10, label='States'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FBBC05', markersize=10, label='Types')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            
            # Save or show
            if output_file:
                plt.tight_layout()
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Graph saved to {output_file}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error generating graph: {e}")
            raise
    
    def visualize_interactive(self, 
                           max_nodes: int = 100, 
                           max_edges: int = 200,
                           output_file: str = 'knowledge_graph.html') -> None:
        """
        Generate an interactive visualization using Pyvis.
        
        Args:
            max_nodes: Maximum number of nodes to display
            max_edges: Maximum number of edges to display
            output_file: Output HTML file path
        """
        if not PYWIS_AVAILABLE:
            self.logger.error("Pyvis not installed. Please install with: pip install pyvis")
            return
            
        if len(self.graph.nodes) == 0:
            self.logger.warning("Cannot visualize empty graph")
            return
            
        start_time = time.time()
        self.logger.info("Generating interactive visualization...")
        
        # Sample the graph if it's too large
        graph_to_plot = self._sample_graph(max_nodes, max_edges)
        self.logger.info(f"Graph sampled to {len(graph_to_plot.nodes)} nodes and {len(graph_to_plot.edges)} edges")
        
        # Create a Pyvis network
        net = Network(
            height="800px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=False
        )
        
        # Add nodes with styling
        for node, node_attrs in graph_to_plot.nodes(data=True):
            node_type = node_attrs.get('node_type', 'other')
            
            # Create a more informative label
            label = node.split('_')[-1]
            if 'model' in node_attrs and 'manufacturer' in node_attrs:
                label = f"{node_attrs['manufacturer']} {node_attrs['model']}"
            elif 'model' in node_attrs:
                label = node_attrs['model']
            elif 'manufacturer' in node_attrs:
                label = node_attrs['manufacturer']
            elif 'state' in node_attrs:
                label = node_attrs['state']
            
            # Create a tooltip with more information
            title = f"<b>Type:</b> {node_type}<br>"
            for key, value in node_attrs.items():
                if key != 'node_type':
                    title += f"<b>{key}:</b> {value}<br>"
            
            # Get node color based on type
            node_color = self._get_node_color(node_type)
            
            net.add_node(
                node,
                label=label,
                title=title,
                color=node_color,
                size=10 + 2 * np.log1p(graph_to_plot.degree(node)),
                font={'size': 12, 'face': 'Arial'}
            )
        
        # Add edges
        for u, v, data in graph_to_plot.edges(data=True):
            edge_title = data.get('type', 'related')
            net.add_edge(u, v, title=edge_title, width=0.5)
        
        # Configure physics for better layout
        net.force_atlas_2based(
            gravity=-50,
            central_gravity=0.01,
            spring_length=100,
            spring_strength=0.08,
            damping=0.4,
            overlap=0.9
        )
        
        # Save to HTML file
        net.show(output_file, notebook=False)
        self.logger.info(f"Interactive visualization saved to {output_file}")
        self.logger.info(f"Interactive visualization generated in {time.time() - start_time:.2f} seconds")
        
        # Open in default browser
        try:
            import webbrowser
            webbrowser.open(f'file://{Path(output_file).absolute()}')
        except Exception as e:
            self.logger.warning(f"Could not open browser: {e}")

    def get_entity_relationships(self, entity_id: str) -> List[Dict]:
        """Get all relationships for a given entity."""
        if entity_id not in self.graph:
            return []
            
        relationships = []
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, neighbor)
            relationships.append({
                'target': neighbor,
                'type': edge_data.get('type', 'related'),
                'attributes': {k: v for k, v in edge_data.items() if k != 'type'}
            })
            
        return relationships

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate and return various statistics about the knowledge graph."""
        if len(self.graph.nodes) == 0:
            return {}
            
        degrees = [d for n, d in self.graph.degree()]
        
        return {
            'num_nodes': len(self.graph.nodes),
            'num_edges': len(self.graph.edges),
            'density': nx.density(self.graph),
            'average_degree': sum(degrees) / len(degrees) if degrees else 0,
            'connected_components': nx.number_connected_components(self.graph),
            'is_dag': nx.is_directed_acyclic_graph(self.graph) if self.graph.is_directed() else None
        }
