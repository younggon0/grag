#!/usr/bin/env python3
"""
Neo4j Utility Script
Provides various Neo4j database management functions
"""

import os
import sys
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
from typing import Optional
import argparse

# Load environment variables
load_dotenv()

class Neo4jUtils:
    def __init__(self):
        """Initialize Neo4j connection"""
        self.uri = os.getenv('NEO4J_URI')
        self.username = os.getenv('NEO4J_USERNAME')
        self.password = os.getenv('NEO4J_PASSWORD')
        self.driver = None
        
        if not all([self.uri, self.username, self.password]):
            print("❌ Error: Neo4j credentials not found in .env file")
            sys.exit(1)
        
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"✅ Connected to Neo4j at {self.uri}")
        except Exception as e:
            print(f"❌ Error connecting to Neo4j: {str(e)}")
            sys.exit(1)
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def get_stats(self):
        """Get database statistics"""
        with self.driver.session() as session:
            # Count nodes by type
            node_types = session.run("""
                MATCH (n:Entity)
                RETURN n.type as type, count(n) as count
                ORDER BY count DESC
            """).data()
            
            # Count relationships
            rel_types = session.run("""
                MATCH ()-[r:RELATES_TO]->()
                RETURN r.type as type, count(r) as count
                ORDER BY count DESC
            """).data()
            
            # Total counts
            total_nodes = session.run("MATCH (n) RETURN count(n) as count").single()['count']
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']
            
            print("\n📊 Database Statistics")
            print("=" * 50)
            print(f"Total Nodes: {total_nodes}")
            print(f"Total Relationships: {total_rels}")
            
            if node_types:
                print("\n📌 Entity Types:")
                for item in node_types:
                    print(f"  • {item['type'] or 'UNKNOWN'}: {item['count']}")
            
            if rel_types:
                print("\n🔗 Relationship Types:")
                for item in rel_types[:10]:  # Top 10
                    print(f"  • {item['type']}: {item['count']}")
    
    def search_entities(self, query: str):
        """Search for entities by name"""
        with self.driver.session() as session:
            results = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($query)
                RETURN e.name as name, e.type as type, e.description as description
                LIMIT 20
            """, query=query).data()
            
            print(f"\n🔍 Search Results for '{query}':")
            print("=" * 50)
            
            if not results:
                print("No entities found")
            else:
                for i, entity in enumerate(results, 1):
                    print(f"\n{i}. {entity['name']} ({entity['type'] or 'UNKNOWN'})")
                    if entity['description']:
                        desc = entity['description'][:100] + '...' if len(entity['description']) > 100 else entity['description']
                        print(f"   {desc}")
    
    def get_entity_details(self, entity_name: str):
        """Get detailed information about an entity"""
        with self.driver.session() as session:
            # Get entity info
            entity = session.run("""
                MATCH (e:Entity {name: $name})
                RETURN e.name as name, e.type as type, e.description as description
            """, name=entity_name).single()
            
            if not entity:
                print(f"❌ Entity '{entity_name}' not found")
                return
            
            print(f"\n📋 Entity: {entity['name']}")
            print("=" * 50)
            print(f"Type: {entity['type'] or 'UNKNOWN'}")
            if entity['description']:
                print(f"Description: {entity['description']}")
            
            # Get relationships
            outgoing = session.run("""
                MATCH (e:Entity {name: $name})-[r:RELATES_TO]->(target)
                RETURN r.type as type, target.name as target, r.description as description
            """, name=entity_name).data()
            
            incoming = session.run("""
                MATCH (source)-[r:RELATES_TO]->(e:Entity {name: $name})
                RETURN r.type as type, source.name as source, r.description as description
            """, name=entity_name).data()
            
            if outgoing:
                print("\n→ Outgoing Relationships:")
                for rel in outgoing:
                    print(f"  • {rel['type']} → {rel['target']}")
                    if rel['description']:
                        print(f"    ({rel['description'][:50]}...)")
            
            if incoming:
                print("\n← Incoming Relationships:")
                for rel in incoming:
                    print(f"  • {rel['source']} → {rel['type']}")
                    if rel['description']:
                        print(f"    ({rel['description'][:50]}...)")
    
    def export_to_json(self, filename: str = "neo4j_export.json"):
        """Export entire graph to JSON"""
        with self.driver.session() as session:
            # Get all entities
            entities = session.run("""
                MATCH (e:Entity)
                RETURN e.name as name, e.type as type, e.description as description
            """).data()
            
            # Get all relationships
            relationships = session.run("""
                MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
                RETURN source.name as source, target.name as target, 
                       r.type as type, r.description as description
            """).data()
            
            export_data = {
                'entities': entities,
                'relationships': relationships,
                'metadata': {
                    'total_entities': len(entities),
                    'total_relationships': len(relationships),
                    'exported_from': self.uri
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"✅ Exported {len(entities)} entities and {len(relationships)} relationships to {filename}")
    
    def clear_database(self, confirm: bool = False):
        """Clear all data from database"""
        with self.driver.session() as session:
            if not confirm:
                # Get current counts
                node_count = session.run("MATCH (n) RETURN count(n) as count").single()['count']
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']
                
                if node_count == 0:
                    print("✅ Database is already empty")
                    return
                
                print(f"⚠️  This will delete {node_count} nodes and {rel_count} relationships")
                response = input("Are you sure? (yes/no): ").lower().strip()
                if response != 'yes':
                    print("❌ Operation cancelled")
                    return
            
            session.run("MATCH (n) DETACH DELETE n")
            print("✅ Database cleared successfully")
    
    def list_recent_entities(self, limit: int = 20):
        """List recently added entities"""
        with self.driver.session() as session:
            results = session.run("""
                MATCH (e:Entity)
                WHERE e.updated IS NOT NULL
                RETURN e.name as name, e.type as type, e.updated as updated
                ORDER BY e.updated DESC
                LIMIT $limit
            """, limit=limit).data()
            
            print(f"\n🕐 Recent Entities (Last {limit}):")
            print("=" * 50)
            
            if not results:
                print("No entities with timestamps found")
            else:
                for entity in results:
                    print(f"• {entity['name']} ({entity['type'] or 'UNKNOWN'})")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Neo4j Database Utilities')
    parser.add_argument('command', choices=['stats', 'search', 'details', 'export', 'clear', 'recent'],
                       help='Command to execute')
    parser.add_argument('--query', '-q', help='Search query or entity name')
    parser.add_argument('--output', '-o', help='Output filename for export')
    parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation prompts')
    parser.add_argument('--limit', '-l', type=int, default=20, help='Limit for recent entities')
    
    args = parser.parse_args()
    
    # Initialize utilities
    utils = Neo4jUtils()
    
    try:
        if args.command == 'stats':
            utils.get_stats()
        
        elif args.command == 'search':
            if not args.query:
                print("❌ Please provide a search query with --query")
                sys.exit(1)
            utils.search_entities(args.query)
        
        elif args.command == 'details':
            if not args.query:
                print("❌ Please provide an entity name with --query")
                sys.exit(1)
            utils.get_entity_details(args.query)
        
        elif args.command == 'export':
            filename = args.output or 'neo4j_export.json'
            utils.export_to_json(filename)
        
        elif args.command == 'clear':
            utils.clear_database(confirm=args.force)
        
        elif args.command == 'recent':
            utils.list_recent_entities(args.limit)
    
    finally:
        utils.close()

if __name__ == "__main__":
    main()