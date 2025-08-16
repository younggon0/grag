#!/usr/bin/env python3
"""
Integration test for URL ingestion with LlamaIndex
"""

import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from components.document_processor import DocumentProcessor
from urllib.parse import urlparse
from datetime import datetime

# Load environment variables
load_dotenv()

# Setup asyncio for LlamaIndex
nest_asyncio.apply()

def test_url_to_graph(url: str):
    """Test converting URL content to knowledge graph"""
    
    print(f"\n🧪 Testing URL to Knowledge Graph Pipeline")
    print(f"🌐 URL: {url}")
    print("=" * 60)
    
    # Initialize components
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Please set ANTHROPIC_API_KEY in .env file")
        return
    
    # Initialize LLM and embeddings
    print("🔧 Initializing LlamaIndex components...")
    llm = Anthropic(
        api_key=api_key,
        model="claude-3-haiku-20240307",
        temperature=0.1
    )
    
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Fetch URL content
    print("📥 Fetching URL content...")
    processor = DocumentProcessor()
    
    try:
        content = processor.fetch_url(url)
        print(f"✅ Fetched {len(content):,} characters")
        
        # Create document with URL metadata
        parsed_url = urlparse(url)
        metadata = {
            "source": url,
            "source_type": "url",
            "domain": parsed_url.netloc,
            "fetch_date": datetime.now().isoformat(),
            "url": url,
            "filename": parsed_url.netloc
        }
        
        document = Document(
            text=content,
            metadata=metadata
        )
        
        # Create knowledge graph extractor
        print("\n🔍 Extracting entities and relationships...")
        kg_extractor = SimpleLLMPathExtractor(
            llm=llm,
            max_paths_per_chunk=5,  # Reduced for testing
            num_workers=1
        )
        
        # Check for Neo4j connection
        neo4j_uri = os.getenv('NEO4J_URI')
        neo4j_user = os.getenv('NEO4J_USERNAME')
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        
        graph_store = None
        if neo4j_uri and neo4j_user and neo4j_password:
            try:
                print("🗄️ Connecting to Neo4j...")
                graph_store = Neo4jPropertyGraphStore(
                    url=neo4j_uri,
                    username=neo4j_user,
                    password=neo4j_password
                )
                print("✅ Connected to Neo4j")
            except Exception as e:
                print(f"⚠️ Neo4j connection failed: {str(e)[:50]}...")
                print("📝 Continuing with in-memory graph")
        
        # Build index
        print("\n🏗️ Building knowledge graph index...")
        index = PropertyGraphIndex.from_documents(
            [document],
            property_graph_store=graph_store,
            kg_extractors=[kg_extractor],
            show_progress=True
        )
        
        print("✅ Knowledge graph created!")
        
        # Get some statistics
        if graph_store:
            try:
                triplets = graph_store.get_triplets()
                print(f"\n📊 Graph Statistics:")
                print(f"   - Total relationships: {len(triplets)}")
                
                # Show first few relationships
                if triplets:
                    print(f"\n📝 Sample Relationships (first 3):")
                    for i, triplet in enumerate(triplets[:3], 1):
                        if isinstance(triplet, (list, tuple)) and len(triplet) == 3:
                            subj_node, rel_node, obj_node = triplet
                            
                            # Extract names
                            subj = subj_node.name if hasattr(subj_node, 'name') else str(subj_node)
                            obj = obj_node.name if hasattr(obj_node, 'name') else str(obj_node)
                            rel = rel_node.label if hasattr(rel_node, 'label') else str(rel_node)
                            
                            print(f"   {i}. {subj} -> {rel} -> {obj}")
            except Exception as e:
                print(f"⚠️ Could not retrieve graph statistics: {str(e)}")
        
        # Test query engine
        print("\n🤖 Testing Q&A capabilities...")
        query_engine = index.as_query_engine(
            include_text=True,
            response_mode="tree_summarize",
            verbose=False
        )
        
        # Ask a simple question
        test_question = "What is the main topic of this content?"
        print(f"   Q: {test_question}")
        
        response = query_engine.query(test_question)
        print(f"   A: {response.response[:200]}...")
        
        # Check source attribution
        if hasattr(response, 'source_nodes') and response.source_nodes:
            print(f"\n📚 Source Attribution:")
            for node in response.source_nodes[:2]:
                if node.metadata:
                    print(f"   - Source: {node.metadata.get('source', 'Unknown')}")
                    print(f"   - Type: {node.metadata.get('source_type', 'Unknown')}")
                    if 'domain' in node.metadata:
                        print(f"   - Domain: {node.metadata.get('domain')}")
        
        print("\n✨ URL ingestion test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test with a simple page
    test_url = "https://en.wikipedia.org/wiki/Knowledge_graph"
    
    import sys
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    
    success = test_url_to_graph(test_url)
    
    if not success:
        sys.exit(1)