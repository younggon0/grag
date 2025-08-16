"""
Graph Builder Module
Builds and manages the NetworkX graph
"""

import networkx as nx
from typing import List, Dict, Any
from difflib import SequenceMatcher

class GraphBuilder:
    def __init__(self, similarity_threshold=0.85):
        """
        Initialize graph builder
        
        Args:
            similarity_threshold: Threshold for entity deduplication
        """
        self.similarity_threshold = similarity_threshold
    
    def add_to_graph(self, graph: nx.DiGraph, entities: List[Dict], relationships: List[Dict]):
        """
        Add entities and relationships to the graph
        
        Args:
            graph: NetworkX graph
            entities: List of entities
            relationships: List of relationships
        """
        # Add entities as nodes
        for entity in entities:
            node_id = entity['name']
            
            # Check if node already exists
            if graph.has_node(node_id):
                # Update attributes if new info is longer/better
                existing_attrs = graph.nodes[node_id]
                if len(entity.get('description', '')) > len(existing_attrs.get('description', '')):
                    graph.nodes[node_id]['description'] = entity['description']
            else:
                # Add new node
                graph.add_node(
                    node_id,
                    type=entity['type'],
                    description=entity.get('description', ''),
                    label=entity['name']
                )
        
        # Add relationships as edges
        for rel in relationships:
            source = rel['source']
            target = rel['target']
            
            # Only add edge if both nodes exist
            if source in graph and target in graph:
                # Check if edge already exists
                if graph.has_edge(source, target):
                    # Add to relationship types if different
                    existing_types = graph[source][target].get('types', [])
                    if rel['type'] not in existing_types:
                        existing_types.append(rel['type'])
                        graph[source][target]['types'] = existing_types
                        graph[source][target]['label'] = ', '.join(existing_types)
                else:
                    # Add new edge
                    graph.add_edge(
                        source,
                        target,
                        type=rel['type'],
                        types=[rel['type']],
                        description=rel.get('description', ''),
                        label=rel['type']
                    )
    
    def deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Deduplicate entities based on name similarity
        
        Args:
            entities: List of entities
            
        Returns:
            Deduplicated list of entities
        """
        if not entities:
            return []
        
        deduplicated = []
        seen_names = {}
        
        for entity in entities:
            name = entity['name'].lower().strip()
            
            # Check for similar existing entities
            is_duplicate = False
            for seen_name, seen_entity in seen_names.items():
                similarity = self._string_similarity(name, seen_name)
                
                if similarity >= self.similarity_threshold:
                    # Merge with existing entity (keep longer description)
                    if len(entity.get('description', '')) > len(seen_entity.get('description', '')):
                        seen_entity['description'] = entity['description']
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_names[name] = entity
                deduplicated.append(entity)
        
        return deduplicated
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate string similarity ratio
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity ratio (0-1)
        """
        return SequenceMatcher(None, s1, s2).ratio()
    
    def get_subgraph(self, graph: nx.DiGraph, node: str, depth: int = 2) -> nx.DiGraph:
        """
        Get a subgraph around a specific node
        
        Args:
            graph: NetworkX graph
            node: Center node
            depth: How many hops from the center node
            
        Returns:
            Subgraph
        """
        if not graph.has_node(node):
            return nx.DiGraph()
        
        # Get nodes within depth hops
        nodes = {node}
        current_level = {node}
        
        for _ in range(depth):
            next_level = set()
            for n in current_level:
                # Add predecessors and successors
                next_level.update(graph.predecessors(n))
                next_level.update(graph.successors(n))
            nodes.update(next_level)
            current_level = next_level
        
        # Create subgraph
        return graph.subgraph(nodes).copy()
    
    def get_graph_stats(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Get statistics about the graph
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary with graph statistics
        """
        stats = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'num_components': nx.number_weakly_connected_components(graph),
            'density': nx.density(graph) if graph.number_of_nodes() > 0 else 0,
            'is_dag': nx.is_directed_acyclic_graph(graph)
        }
        
        if graph.number_of_nodes() > 0:
            # Get most connected nodes
            degrees = dict(graph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            stats['top_nodes'] = [{'name': name, 'degree': deg} for name, deg in top_nodes]
            
            # Get entity type distribution
            type_counts = {}
            for node, attrs in graph.nodes(data=True):
                node_type = attrs.get('type', 'UNKNOWN')
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
            stats['entity_types'] = type_counts
        
        return stats