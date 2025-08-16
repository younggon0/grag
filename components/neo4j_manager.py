"""
Neo4j Graph Manager
Handles persistent storage and retrieval of knowledge graphs in Neo4j
"""

import os
from neo4j import GraphDatabase
import networkx as nx
from typing import List, Dict, Any, Optional
import streamlit as st

class Neo4jManager:
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j URI (defaults to env variable)
            username: Neo4j username (defaults to env variable)
            password: Neo4j password (defaults to env variable)
        """
        self.uri = uri or os.getenv('NEO4J_URI')
        self.username = username or os.getenv('NEO4J_USERNAME')
        self.password = password or os.getenv('NEO4J_PASSWORD')
        
        self.driver = None
        if self.uri and self.username and self.password:
            try:
                self.driver = GraphDatabase.driver(
                    self.uri, 
                    auth=(self.username, self.password)
                )
                # Test connection
                with self.driver.session() as session:
                    session.run("RETURN 1")
                self.connected = True
            except Exception as e:
                st.warning(f"Neo4j connection failed: {str(e)}. Using in-memory graph only.")
                self.connected = False
        else:
            self.connected = False
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def clear_graph(self):
        """Clear all nodes and relationships from Neo4j"""
        if not self.connected:
            return False
        
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            return True
        except Exception as e:
            st.error(f"Error clearing Neo4j graph: {str(e)}")
            return False
    
    def save_to_neo4j(self, entities: List[Dict], relationships: List[Dict]):
        """
        Save entities and relationships to Neo4j
        
        Args:
            entities: List of entity dictionaries
            relationships: List of relationship dictionaries
        """
        if not self.connected:
            return False
        
        try:
            with self.driver.session() as session:
                # Create or merge entities
                for entity in entities:
                    query = """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type, 
                        e.description = $description,
                        e.updated = timestamp()
                    """
                    session.run(query, 
                              name=entity['name'],
                              type=entity.get('type', 'UNKNOWN'),
                              description=entity.get('description', ''))
                
                # Create relationships
                for rel in relationships:
                    query = """
                    MATCH (source:Entity {name: $source})
                    MATCH (target:Entity {name: $target})
                    MERGE (source)-[r:RELATES_TO {type: $rel_type}]->(target)
                    SET r.description = $description,
                        r.updated = timestamp()
                    """
                    session.run(query,
                              source=rel['source'],
                              target=rel['target'],
                              rel_type=rel.get('type', 'relates_to'),
                              description=rel.get('description', ''))
            
            return True
        except Exception as e:
            st.error(f"Error saving to Neo4j: {str(e)}")
            return False
    
    def load_from_neo4j(self) -> Dict[str, Any]:
        """
        Load graph data from Neo4j
        
        Returns:
            Dictionary with entities and relationships
        """
        if not self.connected:
            return {'entities': [], 'relationships': []}
        
        try:
            entities = []
            relationships = []
            
            with self.driver.session() as session:
                # Load entities
                entity_result = session.run("""
                    MATCH (e:Entity)
                    RETURN e.name as name, e.type as type, e.description as description
                """)
                
                for record in entity_result:
                    entities.append({
                        'name': record['name'],
                        'type': record['type'] or 'UNKNOWN',
                        'description': record['description'] or ''
                    })
                
                # Load relationships
                rel_result = session.run("""
                    MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
                    RETURN source.name as source, target.name as target, 
                           r.type as type, r.description as description
                """)
                
                for record in rel_result:
                    relationships.append({
                        'source': record['source'],
                        'target': record['target'],
                        'type': record['type'] or 'relates_to',
                        'description': record['description'] or ''
                    })
            
            return {
                'entities': entities,
                'relationships': relationships
            }
        except Exception as e:
            st.error(f"Error loading from Neo4j: {str(e)}")
            return {'entities': [], 'relationships': []}
    
    def neo4j_to_networkx(self) -> nx.DiGraph:
        """
        Convert Neo4j graph to NetworkX DiGraph
        
        Returns:
            NetworkX DiGraph
        """
        graph = nx.DiGraph()
        data = self.load_from_neo4j()
        
        # Add nodes
        for entity in data['entities']:
            graph.add_node(
                entity['name'],
                type=entity['type'],
                description=entity['description'],
                label=entity['name']
            )
        
        # Add edges
        for rel in data['relationships']:
            if graph.has_node(rel['source']) and graph.has_node(rel['target']):
                graph.add_edge(
                    rel['source'],
                    rel['target'],
                    type=rel['type'],
                    description=rel['description'],
                    label=rel['type']
                )
        
        return graph
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get graph statistics from Neo4j
        
        Returns:
            Dictionary with statistics
        """
        if not self.connected:
            return {'nodes': 0, 'relationships': 0}
        
        try:
            with self.driver.session() as session:
                node_count = session.run("MATCH (n) RETURN count(n) as count").single()['count']
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']
                
                return {
                    'nodes': node_count,
                    'relationships': rel_count
                }
        except Exception as e:
            st.error(f"Error getting statistics: {str(e)}")
            return {'nodes': 0, 'relationships': 0}
    
    def search_entities(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for entities in Neo4j
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching entities
        """
        if not self.connected:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($query)
                       OR toLower(e.description) CONTAINS toLower($query)
                    RETURN e.name as name, e.type as type, e.description as description
                    LIMIT $limit
                """, query=query, limit=limit)
                
                return [dict(record) for record in result]
        except Exception as e:
            st.error(f"Error searching entities: {str(e)}")
            return []
    
    def get_entity_neighbors(self, entity_name: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get neighbors of an entity up to specified depth
        
        Args:
            entity_name: Name of the entity
            depth: How many hops from the entity
            
        Returns:
            Dictionary with entities and relationships
        """
        if not self.connected:
            return {'entities': [], 'relationships': []}
        
        try:
            with self.driver.session() as session:
                # Get connected entities up to depth
                result = session.run("""
                    MATCH (start:Entity {name: $name})
                    CALL apoc.neighbors.athop(start, "RELATES_TO", $depth)
                    YIELD node
                    WITH collect(DISTINCT node) as nodes, start
                    MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
                    WHERE source IN nodes + [start] AND target IN nodes + [start]
                    RETURN 
                        collect(DISTINCT {
                            name: source.name, 
                            type: source.type, 
                            description: source.description
                        }) + collect(DISTINCT {
                            name: target.name, 
                            type: target.type, 
                            description: target.description
                        }) as entities,
                        collect(DISTINCT {
                            source: source.name, 
                            target: target.name, 
                            type: r.type, 
                            description: r.description
                        }) as relationships
                """, name=entity_name, depth=depth)
                
                record = result.single()
                if record:
                    # Deduplicate entities
                    entities = {e['name']: e for e in record['entities']}.values()
                    return {
                        'entities': list(entities),
                        'relationships': record['relationships']
                    }
                    
        except Exception as e:
            # Fallback if APOC is not available
            try:
                with self.driver.session() as session:
                    result = session.run("""
                        MATCH path = (start:Entity {name: $name})-[:RELATES_TO*0..""" + str(depth) + """]-()
                        UNWIND nodes(path) as node
                        WITH collect(DISTINCT node) as nodes
                        MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
                        WHERE source IN nodes AND target IN nodes
                        RETURN 
                            collect(DISTINCT {
                                name: source.name, 
                                type: source.type, 
                                description: source.description
                            }) + collect(DISTINCT {
                                name: target.name, 
                                type: target.type, 
                                description: target.description
                            }) as entities,
                            collect(DISTINCT {
                                source: source.name, 
                                target: target.name, 
                                type: r.type, 
                                description: r.description
                            }) as relationships
                    """, name=entity_name)
                    
                    record = result.single()
                    if record:
                        # Deduplicate entities
                        entities = {e['name']: e for e in record['entities']}.values()
                        return {
                            'entities': list(entities),
                            'relationships': record['relationships']
                        }
            except:
                pass
                
        return {'entities': [], 'relationships': []}