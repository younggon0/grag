"""
Universal Knowledge Graph Builder v2
Using LlamaIndex for document-aware knowledge graphs with source attribution
"""

import os
import streamlit as st
import networkx as nx
from dotenv import load_dotenv
import json
import tempfile
from pathlib import Path
from typing import List, Optional
import asyncio
import nest_asyncio

# LlamaIndex imports
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

# Visualization
from pyvis.network import Network
import pandas as pd

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Knowledge Graph Builder v2",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'index' not in st.session_state:
    st.session_state.index = None
    st.session_state.documents = []
    st.session_state.graph_store = None
    st.session_state.query_engine = None
    st.session_state.existing_data_loaded = False
    st.session_state.current_content = None
    st.session_state.current_source = None
    st.session_state.neo4j_status = 'unknown'
    st.session_state.neo4j_message = 'Initializing...'

@st.cache_resource
def init_llama_index():
    """Initialize LlamaIndex components"""
    
    # Fix asyncio event loop issue for Streamlit
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    # Apply nest_asyncio to allow nested event loops
    nest_asyncio.apply()
    
    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        st.error("⚠️ Please set your ANTHROPIC_API_KEY in the .env file")
        st.stop()
    
    # Initialize LLM
    llm = Anthropic(
        api_key=api_key,
        model="claude-3-haiku-20240307",
        temperature=0.1
    )
    
    # Initialize embedding model (using free HuggingFace model)
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 200
    
    # Initialize Neo4j store if credentials available
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_user = os.getenv('NEO4J_USERNAME')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    graph_store = None
    neo4j_status = "disconnected"
    neo4j_message = "No Neo4j credentials"
    
    if neo4j_uri and neo4j_user and neo4j_password:
        try:
            graph_store = Neo4jPropertyGraphStore(
                url=neo4j_uri,  # Changed from 'uri' to 'url'
                username=neo4j_user,
                password=neo4j_password
            )
            neo4j_status = "connected"
            neo4j_message = "Connected to Neo4j"
        except Exception as e:
            neo4j_status = "error"
            neo4j_message = f"Connection failed: {str(e)[:50]}..."
    
    # Store status in session state for display
    st.session_state.neo4j_status = neo4j_status
    st.session_state.neo4j_message = neo4j_message
    
    return llm, embed_model, graph_store

def load_existing_data_from_neo4j():
    """Load existing data from Neo4j if available"""
    if st.session_state.existing_data_loaded:
        return
    
    try:
        # Ensure we have event loop set up
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                asyncio.set_event_loop(asyncio.new_event_loop())
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        
        llm, embed_model, graph_store = init_llama_index()
        
        if graph_store:
            # Check if there's existing data
            triplets = graph_store.get_triplets()
            
            if triplets and len(triplets) > 0:
                # Create an index with the existing graph store
                st.session_state.index = PropertyGraphIndex(
                    nodes=[],  # Empty nodes since we're using existing data
                    property_graph_store=graph_store
                )
                
                # Create query engine
                st.session_state.query_engine = st.session_state.index.as_query_engine(
                    include_text=True,
                    response_mode="tree_summarize",
                    verbose=True
                )
                
                # Try to populate documents list from existing data
                try:
                    query = "MATCH (n) WHERE n.source IS NOT NULL RETURN DISTINCT n.source AS source"
                    result = graph_store.structured_query(query)
                    if result:
                        existing_sources = [row['source'] for row in result]
                        st.session_state.documents.extend(existing_sources)
                        st.session_state.documents = list(set(st.session_state.documents))  # Remove duplicates
                except:
                    pass
                
                st.session_state.existing_data_loaded = True
                # Store loaded data info in session state instead of showing prominent message
                st.session_state.loaded_relationships = len(triplets)
                
    except Exception as e:
        # Store error in session state instead of showing warning
        st.session_state.neo4j_load_error = str(e)
    
    st.session_state.existing_data_loaded = True

def process_document(content: str, source_name: str, index: Optional[PropertyGraphIndex] = None):
    """Process a document and add to knowledge graph"""
    
    # Create Document object
    document = Document(
        text=content,
        metadata={
            "source": source_name,
            "filename": source_name
        }
    )
    
    # Get components
    llm, embed_model, graph_store = init_llama_index()
    
    # Create extractor
    kg_extractor = SimpleLLMPathExtractor(
        llm=llm,
        max_paths_per_chunk=10,
        num_workers=4
    )
    
    # Create or update index
    if index is None:
        # Create new index
        with st.spinner("🔍 Extracting entities and relationships..."):
            index = PropertyGraphIndex.from_documents(
                [document],
                property_graph_store=graph_store,
                kg_extractors=[kg_extractor],
                show_progress=True
            )
    else:
        # Add to existing index using the insert method
        with st.spinner("🔍 Adding to knowledge graph..."):
            try:
                # Use the insert method to add a single document
                index.insert(document)
            except Exception as e:
                st.error(f"Error adding document to existing index: {str(e)}")
                # Fallback: create new index
                index = PropertyGraphIndex.from_documents(
                    [document],
                    property_graph_store=graph_store,
                    kg_extractors=[kg_extractor],
                    show_progress=True
                )
    
    return index

def create_pyvis_graph(index: PropertyGraphIndex) -> str:
    """Create interactive Pyvis visualization from PropertyGraphIndex"""
    
    # Get the property graph
    property_graph = index.property_graph_store
    
    # Create Pyvis network
    net = Network(
        height="600px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#ffffff",
        font_color="#000000"
    )
    
    # Configure physics
    net.barnes_hut(
        gravity=-80000,
        central_gravity=0.3,
        spring_length=100,
        spring_strength=0.001,
        damping=0.09
    )
    
    # Color map for entity types
    color_map = {
        'PERSON': '#FF6B6B',
        'ORGANIZATION': '#4ECDC4',
        'LOCATION': '#95E77E',
        'DATE': '#FFA07A',
        'EVENT': '#DDA0DD',
        'DEFAULT': '#45B7D1'
    }
    
    # Get all triplets from the graph
    try:
        # Try to get triplets directly
        if hasattr(property_graph, 'get_triplets'):
            triplets = property_graph.get_triplets()
        else:
            return "<p>Graph visualization not available for this store type</p>"
    except Exception as e:
        st.error(f"Error retrieving graph data: {str(e)}")
        triplets = []
    
    if not triplets:
        return "<p>No graph data to visualize</p>"
    
    # Track unique nodes
    nodes_added = set()
    
    # Add nodes and edges from triplets
    # LlamaIndex triplets are in the format [EntityNode, Relation, EntityNode]
    for triplet in triplets[:50]:  # Limit to 50 for performance
        try:
            if isinstance(triplet, (list, tuple)) and len(triplet) == 3:
                subj_node, rel_node, obj_node = triplet
                
                # Extract subject
                if hasattr(subj_node, 'name'):
                    subj = subj_node.name
                elif hasattr(subj_node, 'id'):
                    subj = subj_node.id
                else:
                    subj = str(subj_node)
                
                # Extract object
                if hasattr(obj_node, 'name'):
                    obj = obj_node.name
                elif hasattr(obj_node, 'id'):
                    obj = obj_node.id
                else:
                    obj = str(obj_node)
                
                # Extract relation
                if hasattr(rel_node, 'label'):
                    rel = rel_node.label
                elif hasattr(rel_node, 'name'):
                    rel = rel_node.name
                else:
                    rel = str(rel_node)
                
                # Clean up the names
                subj = subj.strip()
                obj = obj.strip()
                rel = rel.strip()
                
                if not subj or not obj or not rel:
                    continue
                
                # Add subject node
                if subj not in nodes_added:
                    # Determine color based on entity type if available
                    subj_color = color_map.get('DEFAULT')
                    if hasattr(subj_node, 'properties') and 'type' in subj_node.properties:
                        entity_type = subj_node.properties['type'].upper()
                        subj_color = color_map.get(entity_type, color_map.get('DEFAULT'))
                    
                    net.add_node(
                        subj,
                        label=subj[:20] + "..." if len(subj) > 20 else subj,
                        color=subj_color,
                        size=20,
                        title=f"Entity: {subj}"
                    )
                    nodes_added.add(subj)
                
                # Add object node
                if obj not in nodes_added:
                    # Determine color based on entity type if available
                    obj_color = color_map.get('DEFAULT')
                    if hasattr(obj_node, 'properties') and 'type' in obj_node.properties:
                        entity_type = obj_node.properties['type'].upper()
                        obj_color = color_map.get(entity_type, color_map.get('DEFAULT'))
                    
                    net.add_node(
                        obj,
                        label=obj[:20] + "..." if len(obj) > 20 else obj,
                        color=obj_color,
                        size=20,
                        title=f"Entity: {obj}"
                    )
                    nodes_added.add(obj)
                
                # Add edge
                net.add_edge(
                    subj,
                    obj,
                    label=rel[:15] + "..." if len(rel) > 15 else rel,
                    title=f"{subj} -> {rel} -> {obj}",
                    arrows='to',
                    color={'color': '#888888', 'opacity': 0.6}
                )
                
        except Exception as e:
            # Skip problematic triplets
            continue
    
    if len(nodes_added) == 0:
        return "<p>No valid graph data found to visualize</p>"
    
    # Generate HTML
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as tmp:
        net.save_graph(tmp.name)
        tmp_path = tmp.name
    
    with open(tmp_path, 'r') as f:
        html_content = f.read()
    
    os.unlink(tmp_path)
    return html_content

def main():
    st.title("🧠 Universal Knowledge Graph Builder")
    st.markdown("Build knowledge graphs with document indexing and source attribution using LlamaIndex")
    
    # Initialize components (this sets neo4j_status in session state)
    llm, embed_model, graph_store = init_llama_index()
    
    # Load existing data from Neo4j if available
    load_existing_data_from_neo4j()
    
    # Get Neo4j status from session state (should be set by init_llama_index)
    neo4j_status = getattr(st.session_state, 'neo4j_status', 'unknown')
    neo4j_message = getattr(st.session_state, 'neo4j_message', 'Status unknown')
    
    # Force status update if still unknown
    if neo4j_status == 'unknown':
        if graph_store is not None:
            st.session_state.neo4j_status = 'connected'
            st.session_state.neo4j_message = 'Connected to Neo4j'
        else:
            st.session_state.neo4j_status = 'disconnected'
            st.session_state.neo4j_message = 'No Neo4j connection'
        neo4j_status = st.session_state.neo4j_status
        neo4j_message = st.session_state.neo4j_message
    
    # Status indicator styling
    status_color = {
        'connected': '#28a745',  # green
        'error': '#dc3545',      # red
        'disconnected': '#6c757d' # gray
    }.get(neo4j_status, '#6c757d')
    
    status_icon = {
        'connected': '🟢',
        'error': '🔴', 
        'disconnected': '⚪'
    }.get(neo4j_status, '⚪')
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Input")
        
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Upload File", "Use Sample", "Enter Text"]
        )
        
        # Reset content/source if input method changes
        if input_method == "Upload File":
            # Clear any previous content from other methods
            if st.session_state.current_source and not st.session_state.current_source.endswith('.txt'):
                st.session_state.current_content = None
                st.session_state.current_source = None
                
            uploaded_file = st.file_uploader(
                "Choose a text file",
                type=['txt'],
                help="Upload a .txt file (max 100MB)"
            )
            
            if uploaded_file:
                if uploaded_file.size > 100 * 1024 * 1024:
                    st.error("File too large! Please upload a file smaller than 100MB.")
                else:
                    st.session_state.current_content = uploaded_file.read().decode('utf-8')
                    st.session_state.current_source = uploaded_file.name
        
        elif input_method == "Use Sample":
            sample_docs = {
                "Technology History": "sample_docs/technology_history.txt",
                "Steve Jobs Bio": "sample_docs/steve_jobs_biography.txt",
                "Silicon Valley": "sample_docs/silicon_valley_companies.txt"
            }
            
            selected_sample = st.selectbox("Select sample document:", list(sample_docs.keys()))
            
            if st.button("Load Sample"):
                sample_path = Path(sample_docs[selected_sample])
                if sample_path.exists():
                    try:
                        with open(sample_path, 'r') as f:
                            st.session_state.current_content = f.read()
                        st.session_state.current_source = selected_sample
                        st.success(f"✅ Loaded sample: {selected_sample}")
                    except Exception as e:
                        st.error(f"Error reading sample file: {str(e)}")
                        # Fallback to built-in sample
                        st.session_state.current_content = """Apple Inc. is an American multinational technology company headquartered in Cupertino, California. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 to develop and sell personal computers. The company was incorporated as Apple Computer, Inc. in January 1977, and sales of its computers saw significant momentum and revenue growth. Steve Jobs resigned from Apple in 1985 and founded NeXT, a computer platform development company. Apple acquired NeXT in 1997, and Jobs returned to Apple as CEO. Under his leadership, Apple introduced revolutionary products like the iMac, iPod, iPhone, and iPad. Tim Cook became CEO in August 2011 after Jobs resigned due to health issues."""
                        st.session_state.current_source = "fallback_sample.txt"
                        st.success("✅ Loaded fallback sample content")
                else:
                    st.error(f"Sample file not found: {sample_path}")
                    # Fallback to built-in sample
                    st.session_state.current_content = """Apple Inc. is an American multinational technology company headquartered in Cupertino, California. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 to develop and sell personal computers. The company was incorporated as Apple Computer, Inc. in January 1977, and sales of its computers saw significant momentum and revenue growth. Steve Jobs resigned from Apple in 1985 and founded NeXT, a computer platform development company. Apple acquired NeXT in 1997, and Jobs returned to Apple as CEO. Under his leadership, Apple introduced revolutionary products like the iMac, iPod, iPhone, and iPad. Tim Cook became CEO in August 2011 after Jobs resigned due to health issues."""
                    st.session_state.current_source = "fallback_sample.txt"
                    st.success("✅ Loaded fallback sample content")
            
            # Show current loaded sample
            if st.session_state.current_content and st.session_state.current_source:
                st.info(f"📄 Ready to process: {st.session_state.current_source}")
        
        else:  # Enter Text
            text_input = st.text_area(
                "Enter text:",
                height=200,
                help="Paste or type your text here"
            )
            if text_input:
                st.session_state.current_content = text_input
                st.session_state.current_source = "manual_input.txt"
        
        # Get content and source from session state
        content = st.session_state.current_content
        source_name = st.session_state.current_source
        
        # Process button
        if content and st.button("🚀 Build Knowledge Graph", type="primary"):
            with st.spinner("Processing document..."):
                st.session_state.index = process_document(
                    content, 
                    source_name,
                    st.session_state.index
                )
                
                # Track processed documents
                if source_name not in st.session_state.documents:
                    st.session_state.documents.append(source_name)
                
                # Create query engine
                st.session_state.query_engine = st.session_state.index.as_query_engine(
                    include_text=True,
                    response_mode="tree_summarize",
                    verbose=True
                )
                
                st.success(f"✅ Successfully processed: {source_name}")
                
                # Clear the current content to avoid reprocessing
                st.session_state.current_content = None
                st.session_state.current_source = None
                
                st.balloons()
        
        # Graph statistics
        if st.session_state.index:
            st.divider()
            st.header("📊 Graph Statistics")
            
            # Get triplets count
            if graph_store:
                try:
                    triplets = graph_store.get_triplets()
                    st.metric("Total Relationships", len(triplets))
                except:
                    st.metric("Graph Store", "Active")
            
            # Try to get document count from Neo4j or session state
            doc_count = len(st.session_state.documents) if st.session_state.documents else 0
            
            # If we have a graph store, try to get unique document sources
            if graph_store and doc_count == 0:
                try:
                    # Query for unique document sources in Neo4j
                    query = "MATCH (n) WHERE n.source IS NOT NULL RETURN DISTINCT n.source AS source"
                    result = graph_store.structured_query(query)
                    if result:
                        doc_count = len(result)
                except:
                    # Fallback: estimate from triplets
                    try:
                        triplets = graph_store.get_triplets()
                        if triplets:
                            # Estimate: if we have triplets, we probably have at least 1 document
                            doc_count = max(1, len(triplets) // 10)  # rough estimate
                    except:
                        doc_count = 0
            
            st.metric("Documents Processed", doc_count)
        
        # Clear graph option
        if st.session_state.index:
            st.divider()
            if st.button("🗑️ Clear Graph", type="secondary"):
                st.session_state.index = None
                st.session_state.query_engine = None
                st.session_state.documents = []
                if graph_store:
                    # Clear Neo4j
                    try:
                        graph_store.structured_query("MATCH (n) DETACH DELETE n")
                    except:
                        pass
                st.success("Graph cleared!")
                st.rerun()
        
        # Small Neo4j status at bottom of sidebar
        st.divider()
        with st.expander("ℹ️ Database Status", expanded=False):
            st.write(f"{status_icon} **Neo4j**: {neo4j_status.title()}")
            st.caption(neo4j_message)
            # Debug info (remove after testing)
            st.caption(f"Debug: status={neo4j_status}, has_graph_store={graph_store is not None}")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🌐 Graph Visualization", "❓ Q&A with Sources", "📝 Raw Triplets"])
    
    with tab1:
        if st.session_state.index:
            st.subheader("Interactive Knowledge Graph")
            
            # Generate and display graph
            try:
                html_graph = create_pyvis_graph(st.session_state.index)
                st.components.v1.html(html_graph, height=600, scrolling=True)
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                st.info("The graph may be too large to visualize. Try the Q&A tab instead.")
        else:
            st.info("👆 Upload a document or use a sample to build a knowledge graph")
    
    with tab2:
        st.subheader("Ask Questions with Source Attribution")
        
        if st.session_state.query_engine:
            # Sample questions
            st.markdown("**Sample Questions:**")
            sample_questions = [
                "Who founded Apple?",
                "What companies did Steve Jobs found?",
                "What is the relationship between Apple and NeXT?",
                "Who succeeded Steve Jobs as CEO?"
            ]
            
            cols = st.columns(len(sample_questions))
            for i, q in enumerate(sample_questions):
                if cols[i].button(q, key=f"sample_{i}"):
                    st.session_state.current_question = q
            
            # Question input
            question = st.text_input(
                "Enter your question:",
                value=st.session_state.get('current_question', ''),
                key="question_input"
            )
            
            if question and st.button("Get Answer"):
                with st.spinner("Searching knowledge graph..."):
                    try:
                        # Query the index
                        response = st.session_state.query_engine.query(question)
                        
                        # Display answer
                        st.success("**Answer:**")
                        st.write(response.response)
                        
                        # Display sources
                        if hasattr(response, 'source_nodes') and response.source_nodes:
                            st.divider()
                            st.subheader("📚 Sources")
                            
                            for i, node in enumerate(response.source_nodes, 1):
                                with st.expander(f"Source {i}: {node.metadata.get('source', 'Unknown')}"):
                                    # Show text snippet
                                    st.markdown("**Text:**")
                                    st.write(node.text[:500] + "..." if len(node.text) > 500 else node.text)
                                    
                                    # Show metadata
                                    if node.metadata:
                                        st.markdown("**Metadata:**")
                                        st.json(node.metadata)
                                    
                                    # Show relevance score if available
                                    if hasattr(node, 'score'):
                                        st.metric("Relevance Score", f"{node.score:.3f}")
                    
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
        else:
            st.info("Build a knowledge graph first to enable Q&A")
    
    with tab3:
        st.subheader("Extracted Relationships")
        
        if st.session_state.index:
            try:
                # Get the property graph store
                property_graph = st.session_state.index.property_graph_store
                
                # Get all triplets
                if hasattr(property_graph, 'get_triplets'):
                    triplets = property_graph.get_triplets()
                    
                    if triplets:
                        # Convert to dataframe with proper format handling
                        data = []
                        for triplet in triplets[:100]:  # Limit to 100 for display
                            try:
                                if isinstance(triplet, (list, tuple)) and len(triplet) == 3:
                                    subj_node, rel_node, obj_node = triplet
                                    
                                    # Extract subject
                                    if hasattr(subj_node, 'name'):
                                        subj = subj_node.name
                                    elif hasattr(subj_node, 'id'):
                                        subj = subj_node.id
                                    else:
                                        subj = str(subj_node)
                                    
                                    # Extract object  
                                    if hasattr(obj_node, 'name'):
                                        obj = obj_node.name
                                    elif hasattr(obj_node, 'id'):
                                        obj = obj_node.id
                                    else:
                                        obj = str(obj_node)
                                    
                                    # Extract relation
                                    if hasattr(rel_node, 'label'):
                                        rel = rel_node.label
                                    elif hasattr(rel_node, 'name'):
                                        rel = rel_node.name
                                    else:
                                        rel = str(rel_node)
                                    
                                    data.append({
                                        'Subject': subj.strip(),
                                        'Relationship': rel.strip(),
                                        'Object': obj.strip()
                                    })
                            except Exception:
                                # Skip problematic triplets
                                continue
                        
                        if data:
                            df = pd.DataFrame(data)
                            st.dataframe(df, use_container_width=True)
                            
                            # Show stats
                            st.metric("Total Relationships", len(data))
                            
                            # Download option
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv,
                                file_name="knowledge_graph_triplets.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No valid relationships found in the data")
                    else:
                        st.info("No relationships extracted yet")
                else:
                    st.info("Graph store does not support triplet retrieval")
            except Exception as e:
                st.error(f"Error loading triplets: {str(e)}")
        else:
            st.info("No graph data available")

if __name__ == "__main__":
    main()