#!/usr/bin/env python3
"""
Clear Neo4j Database Script
Removes all nodes and relationships from the Neo4j database
"""

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

def clear_neo4j():
    """Clear all data from Neo4j database"""
    
    # Get Neo4j credentials from environment
    uri = os.getenv('NEO4J_URI')
    username = os.getenv('NEO4J_USERNAME')
    password = os.getenv('NEO4J_PASSWORD')
    
    if not all([uri, username, password]):
        print("❌ Error: Neo4j credentials not found in .env file")
        print("Please ensure NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD are set")
        return False
    
    try:
        # Connect to Neo4j
        print(f"🔗 Connecting to Neo4j at {uri}...")
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Verify connection
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = result.single()['count']
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()['count']
            
            print(f"📊 Current graph contains {node_count} nodes and {rel_count} relationships")
            
            if node_count == 0 and rel_count == 0:
                print("✅ Graph is already empty!")
                driver.close()
                return True
            
            # Confirm before clearing
            print("\n⚠️  WARNING: This will permanently delete all data in the Neo4j database!")
            confirm = input("Are you sure you want to continue? (yes/no): ").lower().strip()
            
            if confirm != 'yes':
                print("❌ Operation cancelled")
                driver.close()
                return False
            
            # Clear all nodes and relationships
            print("\n🗑️  Clearing Neo4j database...")
            session.run("MATCH (n) DETACH DELETE n")
            
            # Verify deletion
            result = session.run("MATCH (n) RETURN count(n) as count")
            new_count = result.single()['count']
            
            if new_count == 0:
                print("✅ Successfully cleared all data from Neo4j!")
                print(f"   Deleted {node_count} nodes and {rel_count} relationships")
            else:
                print(f"⚠️  Warning: {new_count} nodes still remain in the database")
        
        driver.close()
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify Neo4j credentials in .env file")
        print("3. Ensure Neo4j instance is running")
        print("4. Check if the URI format is correct (e.g., neo4j+s://...)")
        return False

def main():
    """Main function"""
    print("=" * 50)
    print("Neo4j Database Cleaner")
    print("=" * 50)
    
    success = clear_neo4j()
    
    if success:
        print("\n✨ Done! You can now start fresh with a clean graph.")
        print("Run 'streamlit run app.py' to start building a new knowledge graph.")
    else:
        print("\n❌ Failed to clear Neo4j database")
        sys.exit(1)

if __name__ == "__main__":
    main()