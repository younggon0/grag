"""
Universal Knowledge Graph Builder
A Streamlit app to convert documents into interactive knowledge graphs with NL Q&A
"""

import os
import streamlit as st
import networkx as nx
from dotenv import load_dotenv
import json
import tempfile
from pathlib import Path

# Load components (to be created)
from components.document_processor import DocumentProcessor
from components.graph_extractor import GraphExtractor
from components.graph_builder import GraphBuilder
from components.visualizer import GraphVisualizer
from components.qa_engine import QAEngine
from components.neo4j_manager import Neo4jManager

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Knowledge Graph Builder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def init_components():
    """Initialize all components with API keys"""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key or api_key == 'your_api_key_here':
        st.error("⚠️ Please set your ANTHROPIC_API_KEY in the .env file")
        st.stop()
    
    # Initialize Neo4j manager
    neo4j_manager = Neo4jManager()
    
    return {
        'processor': DocumentProcessor(),
        'extractor': GraphExtractor(api_key),
        'builder': GraphBuilder(),
        'visualizer': GraphVisualizer(),
        'qa_engine': QAEngine(api_key),
        'neo4j': neo4j_manager
    }

# Initialize session state
if 'graph' not in st.session_state:
    st.session_state.graph = nx.DiGraph()
    st.session_state.processed_docs = []
    st.session_state.extraction_cache = {}
    st.session_state.entities = []
    st.session_state.relationships = []
    
    # Load from Neo4j if connected
    components = init_components()
    if components['neo4j'].connected:
        st.info("🔄 Loading graph from Neo4j...")
        data = components['neo4j'].load_from_neo4j()
        st.session_state.entities = data['entities']
        st.session_state.relationships = data['relationships']
        
        # Rebuild NetworkX graph from Neo4j data
        if data['entities'] or data['relationships']:
            st.session_state.graph = components['neo4j'].neo4j_to_networkx()
            st.success(f"✅ Loaded {len(data['entities'])} entities and {len(data['relationships'])} relationships from Neo4j")

# Main app
def main():
    st.title("🧠 Universal Knowledge Graph Builder")
    st.markdown("Convert documents into interactive knowledge graphs with natural language Q&A")
    
    # Initialize components
    components = init_components()
    
    # Sidebar for input
    with st.sidebar:
        # Neo4j connection status
        if components['neo4j'].connected:
            st.success("✅ Connected to Neo4j")
            stats = components['neo4j'].get_statistics()
            if stats['nodes'] > 0:
                st.info(f"📊 Neo4j: {stats['nodes']} nodes, {stats['relationships']} relationships")
        else:
            st.warning("⚠️ Neo4j not connected - using in-memory graph only")
        
        st.header("📁 Document Input")
        
        input_method = st.radio(
            "Choose input method:",
            ["Upload File", "Enter URL", "Use Sample"]
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
                if uploaded_file.size > 100 * 1024 * 1024:  # 100MB limit
                    st.error("File too large! Please upload a file smaller than 100MB.")
                else:
                    content = uploaded_file.read().decode('utf-8')
                    source_name = uploaded_file.name
                    
        elif input_method == "Enter URL":
            url = st.text_input("Enter URL to process:")
            if st.button("Fetch Content"):
                with st.spinner("Fetching content from URL..."):
                    try:
                        content = components['processor'].fetch_url(url)
                        source_name = url
                    except Exception as e:
                        st.error(f"Error fetching URL: {str(e)}")
                        
        else:  # Use Sample
            if st.button("Load Sample Document"):
                content = """
                Apple Inc. is an American multinational technology company headquartered in Cupertino, California. 
                It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 to develop and sell personal computers.
                The company was incorporated as Apple Computer, Inc. in January 1977, and sales of its computers saw significant momentum and revenue growth.
                
                Steve Jobs resigned from Apple in 1985 and founded NeXT, a computer platform development company. 
                Apple acquired NeXT in 1997, and Jobs returned to Apple as CEO. Under his leadership, Apple introduced revolutionary products 
                like the iMac, iPod, iPhone, and iPad. Tim Cook became CEO in August 2011 after Jobs resigned due to health issues.
                
                Apple's product line includes the iPhone smartphone, iPad tablet computer, Mac personal computer, Apple Watch smartwatch, 
                and Apple TV digital media player. The company also offers services like the App Store, Apple Music, iCloud, and Apple Pay.
                Apple is known for its innovation in consumer electronics and has a strong focus on design and user experience.
                """
                source_name = "sample_apple.txt"
        
        # Process button
        if content and st.button("🚀 Build Knowledge Graph", type="primary"):
            process_document(content, source_name, components)
        
        # Graph statistics
        if st.session_state.graph.number_of_nodes() > 0:
            st.divider()
            st.header("📊 Graph Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nodes", st.session_state.graph.number_of_nodes())
                st.metric("Edges", st.session_state.graph.number_of_edges())
            with col2:
                components_count = nx.number_weakly_connected_components(st.session_state.graph)
                st.metric("Components", components_count)
                st.metric("Docs Processed", len(st.session_state.processed_docs))
            
            # Export options
            st.divider()
            st.header("💾 Export")
            if st.button("Download Graph as JSON"):
                json_data = export_graph_json()
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name="knowledge_graph.json",
                    mime="application/json"
                )
            
            # Clear graph option
            st.divider()
            st.header("⚠️ Manage Graph")
            if st.button("🗑️ Clear All Data", type="secondary"):
                if components['neo4j'].connected:
                    components['neo4j'].clear_graph()
                st.session_state.graph = nx.DiGraph()
                st.session_state.entities = []
                st.session_state.relationships = []
                st.session_state.processed_docs = []
                st.session_state.extraction_cache = {}
                st.success("Graph cleared successfully!")
                st.rerun()
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["🌐 Graph Visualization", "❓ Q&A Interface", "📝 Entities & Relations"])
    
    with tab1:
        if st.session_state.graph.number_of_nodes() > 0:
            st.subheader("Interactive Knowledge Graph")
            
            # Visualization controls
            col1, col2, col3 = st.columns(3)
            with col1:
                physics = st.checkbox("Enable Physics", value=True)
            with col2:
                show_labels = st.checkbox("Show Labels", value=True)
            with col3:
                if st.button("🔄 Refresh View"):
                    st.rerun()
            
            # Generate and display graph
            html_graph = components['visualizer'].create_pyvis_graph(
                st.session_state.graph,
                physics=physics,
                show_labels=show_labels
            )
            
            # Display the graph
            st.components.v1.html(html_graph, height=600, scrolling=True)
        else:
            st.info("👆 Upload a document or enter a URL to build a knowledge graph")
    
    with tab2:
        st.subheader("Ask Questions About the Knowledge Graph")
        
        if st.session_state.graph.number_of_nodes() > 0:
            question = st.text_input("Enter your question:")
            
            if question and st.button("Get Answer"):
                with st.spinner("Thinking..."):
                    answer = components['qa_engine'].answer_question(
                        question,
                        st.session_state.graph,
                        st.session_state.entities,
                        st.session_state.relationships
                    )
                    
                    st.success("**Answer:**")
                    st.write(answer)
        else:
            st.info("Build a knowledge graph first to enable Q&A")
    
    with tab3:
        st.subheader("Extracted Entities and Relationships")
        
        if st.session_state.entities:
            # Entity display
            st.markdown("### 🔵 Entities")
            entity_df = create_entity_dataframe()
            st.dataframe(entity_df, use_container_width=True)
            
            # Relationship display
            if st.session_state.relationships:
                st.markdown("### 🔗 Relationships")
                rel_df = create_relationship_dataframe()
                st.dataframe(rel_df, use_container_width=True)
        else:
            st.info("No entities extracted yet. Process a document to see results.")

def process_document(content, source_name, components):
    """Process a document and build the knowledge graph"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Process document
        status_text.text("📄 Processing document...")
        progress_bar.progress(20)
        chunks = components['processor'].process_text(content)
        
        # Step 2: Extract entities and relationships
        status_text.text("🔍 Extracting entities and relationships...")
        progress_bar.progress(40)
        
        all_entities = []
        all_relationships = []
        
        for i, chunk in enumerate(chunks):
            progress_bar.progress(40 + (40 * i // len(chunks)))
            
            # Check cache
            chunk_hash = hash(chunk)
            if chunk_hash in st.session_state.extraction_cache:
                extraction = st.session_state.extraction_cache[chunk_hash]
            else:
                extraction = components['extractor'].extract(chunk)
                st.session_state.extraction_cache[chunk_hash] = extraction
            
            all_entities.extend(extraction.get('entities', []))
            all_relationships.extend(extraction.get('relationships', []))
        
        # Step 3: Build graph
        status_text.text("🏗️ Building knowledge graph...")
        progress_bar.progress(80)
        
        # Deduplicate entities
        all_entities = components['builder'].deduplicate_entities(all_entities)
        
        # Add to graph
        components['builder'].add_to_graph(
            st.session_state.graph,
            all_entities,
            all_relationships
        )
        
        # Update session state
        st.session_state.entities.extend(all_entities)
        st.session_state.relationships.extend(all_relationships)
        st.session_state.processed_docs.append(source_name)
        
        # Save to Neo4j if connected
        if components['neo4j'].connected:
            status_text.text("💾 Saving to Neo4j...")
            components['neo4j'].save_to_neo4j(all_entities, all_relationships)
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Knowledge graph built successfully!")
        
        st.success(f"Successfully processed: {source_name}")
        st.balloons()
        
        # Auto-refresh to show the graph
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def export_graph_json():
    """Export the graph as JSON"""
    export_data = {
        'entities': st.session_state.entities,
        'relationships': st.session_state.relationships,
        'graph': nx.node_link_data(st.session_state.graph)
    }
    return json.dumps(export_data, indent=2)

def create_entity_dataframe():
    """Create a dataframe from entities"""
    import pandas as pd
    
    data = []
    for entity in st.session_state.entities[:100]:  # Limit to 100 for display
        data.append({
            'Name': entity.get('name', ''),
            'Type': entity.get('type', ''),
            'Description': entity.get('description', '')[:100] + '...' if len(entity.get('description', '')) > 100 else entity.get('description', '')
        })
    
    return pd.DataFrame(data)

def create_relationship_dataframe():
    """Create a dataframe from relationships"""
    import pandas as pd
    
    data = []
    for rel in st.session_state.relationships[:100]:  # Limit to 100 for display
        data.append({
            'Source': rel.get('source', ''),
            'Relationship': rel.get('type', ''),
            'Target': rel.get('target', ''),
            'Description': rel.get('description', '')[:100] + '...' if len(rel.get('description', '')) > 100 else rel.get('description', '')
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    main()