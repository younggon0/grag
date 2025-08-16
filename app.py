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
    if neo4j_uri and neo4j_user and neo4j_password:
        try:
            graph_store = Neo4jPropertyGraphStore(
                url=neo4j_uri,  # Changed from 'uri' to 'url'
                username=neo4j_user,
                password=neo4j_password
            )
            st.success("✅ Connected to Neo4j")
        except Exception as e:
            st.warning(f"⚠️ Neo4j connection failed: {str(e)}. Using in-memory graph.")
    else:
        st.info("ℹ️ Using in-memory graph (no Neo4j credentials found)")
    
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
                
                st.session_state.existing_data_loaded = True
                st.success(f"✅ Loaded existing knowledge graph with {len(triplets)} relationships from Neo4j")
                
    except Exception as e:
        st.warning(f"Could not load existing data from Neo4j: {str(e)}")
    
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
        # Add to existing index
        with st.spinner("🔍 Adding to knowledge graph..."):
            index.insert_documents([document])
    
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
    st.title("🧠 Universal Knowledge Graph Builder v2")
    st.markdown("Build knowledge graphs with document indexing and source attribution using LlamaIndex")
    
    # Initialize components
    llm, embed_model, graph_store = init_llama_index()
    
    # Load existing data from Neo4j if available
    load_existing_data_from_neo4j()
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Input")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Upload File", "Use Sample", "Enter Text"]
        )
        
        content = None
        source_name = None
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a text file",
                type=['txt'],
                help="Upload a .txt file (max 100MB)"
            )
            
            if uploaded_file:
                if uploaded_file.size > 100 * 1024 * 1024:
                    st.error("File too large! Please upload a file smaller than 100MB.")
                else:
                    content = uploaded_file.read().decode('utf-8')
                    source_name = uploaded_file.name
        
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
                    with open(sample_path, 'r') as f:
                        content = f.read()
                    source_name = selected_sample
                else:
                    st.error(f"Sample file not found: {sample_path}")
        
        else:  # Enter Text
            text_input = st.text_area(
                "Enter text:",
                height=200,
                help="Paste or type your text here"
            )
            if text_input:
                content = text_input
                source_name = "manual_input.txt"
        
        # Process button
        if content and st.button("🚀 Build Knowledge Graph", type="primary"):
            with st.spinner("Processing document..."):
                st.session_state.index = process_document(
                    content, 
                    source_name,
                    st.session_state.index
                )
                
                # Create query engine
                st.session_state.query_engine = st.session_state.index.as_query_engine(
                    include_text=True,
                    response_mode="tree_summarize",
                    verbose=True
                )
                
                st.success(f"✅ Successfully processed: {source_name}")
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
            
            st.metric("Documents Processed", len(st.session_state.documents) if hasattr(st.session_state, 'documents') else 1)
        
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