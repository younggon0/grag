# CLAUDE.md - Project Guidelines for Knowledge Graph Builder

## Project Context
Universal Knowledge-Graph Builder using Anthropic Claude API with LlamaIndex for document-aware knowledge graphs with source attribution.

## API Configuration

### Required Environment Variables
```bash
ANTHROPIC_API_KEY=your_api_key_here
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### Model Selection
- **Primary Model**: Claude 3 Haiku (`claude-3-haiku-20240307`)
  - Use for entity/relationship extraction (fast & cheap)
  - Batch processing of document chunks
- **Fallback Model**: Claude 3 Sonnet (`claude-3-sonnet-20240229`)
  - Use for complex Q&A if time permits
  - Better reasoning for graph queries

## Code Standards

### Import Organization
```python
# Standard library
import os
import json
import tempfile
import asyncio
import nest_asyncio
from pathlib import Path
from typing import List, Optional

# Third-party
import streamlit as st
import networkx as nx
from pyvis.network import Network
import pandas as pd
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
```

### Error Handling Pattern
```python
def safe_extract(text: str) -> Dict:
    """Always return valid structure even on failure"""
    try:
        result = extract_entities(text)
        return result
    except Exception as e:
        st.warning(f"Extraction failed for chunk: {str(e)[:100]}")
        return {"entities": [], "relationships": []}
```

### Streamlit Session State
```python
# Initialize once
if 'index' not in st.session_state:
    st.session_state.index = None
    st.session_state.documents = []
    st.session_state.graph_store = None
    st.session_state.query_engine = None
    st.session_state.existing_data_loaded = False
```

## LlamaIndex Patterns

### Initialization with Async Support
```python
@st.cache_resource
def init_llama_index():
    # Fix asyncio event loop issue for Streamlit
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    # Apply nest_asyncio to allow nested event loops
    nest_asyncio.apply()
    
    # Initialize LLM and embedding models
    llm = Anthropic(api_key=api_key, model="claude-3-haiku-20240307", temperature=0.1)
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Initialize Neo4j store
    graph_store = Neo4jPropertyGraphStore(url=neo4j_uri, username=neo4j_user, password=neo4j_password)
    return llm, embed_model, graph_store
```

### Document Processing
```python
def process_document(content: str, source_name: str, index: Optional[PropertyGraphIndex] = None):
    document = Document(text=content, metadata={"source": source_name, "filename": source_name})
    
    llm, embed_model, graph_store = init_llama_index()
    kg_extractor = SimpleLLMPathExtractor(llm=llm, max_paths_per_chunk=10, num_workers=4)
    
    if index is None:
        index = PropertyGraphIndex.from_documents(
            [document], property_graph_store=graph_store, kg_extractors=[kg_extractor]
        )
    else:
        index.insert_documents([document])
    
    return index
```

### Loading Existing Data
```python
def load_existing_data_from_neo4j():
    if st.session_state.existing_data_loaded:
        return
    
    llm, embed_model, graph_store = init_llama_index()
    
    if graph_store:
        triplets = graph_store.get_triplets()
        if triplets and len(triplets) > 0:
            st.session_state.index = PropertyGraphIndex(nodes=[], property_graph_store=graph_store)
            st.session_state.query_engine = st.session_state.index.as_query_engine(
                include_text=True, response_mode="tree_summarize"
            )
            st.success(f"✅ Loaded {len(triplets)} relationships from Neo4j")
    
    st.session_state.existing_data_loaded = True
```

## Performance Guidelines

### Caching Strategy
1. Cache LlamaIndex components with @st.cache_resource
2. Store PropertyGraphIndex in session state
3. Reuse graph store connections across reruns

### Batch Processing
```python
# Process chunks in batches of 5
BATCH_SIZE = 5
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i+BATCH_SIZE]
    # Process batch in parallel if possible
```

### Graph Size Limits
- Max 50 triplets for visualization (performance)
- Limit to first 100 relationships in raw data tab
- Auto-truncate long entity names in visualization

### Graph Visualization with LlamaIndex
```python
def create_pyvis_graph(index: PropertyGraphIndex) -> str:
    property_graph = index.property_graph_store
    triplets = property_graph.get_triplets()
    
    # LlamaIndex triplets are [EntityNode, Relation, EntityNode]
    for triplet in triplets[:50]:  # Limit for performance
        subj_node, rel_node, obj_node = triplet
        
        # Extract names/IDs from nodes
        subj = subj_node.name if hasattr(subj_node, 'name') else subj_node.id
        obj = obj_node.name if hasattr(obj_node, 'name') else obj_node.id
        rel = rel_node.label if hasattr(rel_node, 'label') else str(rel_node)
        
        # Add nodes and edges to Pyvis network
        net.add_node(subj, label=subj[:20])
        net.add_node(obj, label=obj[:20])
        net.add_edge(subj, obj, label=rel[:15])
```

## Quick Command Reference

### Setup Commands
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

### Testing Commands
```bash
# Quick test with sample file
echo "Apple Inc. was founded by Steve Jobs in 1976." > test.txt

# Test entity extraction
python -c "from components.graph_extractor import extract; print(extract('test text'))"
```

## Prompt Engineering Tips

### Entity Extraction Prompt
```python
EXTRACTION_TEMPLATE = """You are a knowledge graph extractor.
Extract entities and relationships from the text below.

Rules:
1. Entity names should be concise (2-4 words max)
2. Entity types: PERSON, ORGANIZATION, CONCEPT, LOCATION, EVENT
3. Relationships should be directional and meaningful
4. Output valid JSON only

Text: {text}

Output JSON:
"""
```

### Q&A Prompt
```python
QA_TEMPLATE = """Given this knowledge graph information:
Entities: {entities}
Relationships: {relationships}

Answer this question concisely: {question}

If the answer is not in the graph, say "Information not found in the knowledge graph."
"""
```

## Common Issues & Solutions

### Issue: Neo4j Connection Failed - Event Loop Error
**Solution**: Fix asyncio event loop for Streamlit
```python
# Add to initialization
try:
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

nest_asyncio.apply()
```

### Issue: Graph Visualization Shows "No data"
**Solution**: Check triplet format handling
```python
# LlamaIndex triplets need proper extraction
for triplet in triplets:
    if isinstance(triplet, (list, tuple)) and len(triplet) == 3:
        subj_node, rel_node, obj_node = triplet
        subj = subj_node.name if hasattr(subj_node, 'name') else subj_node.id
        # Continue processing...
```

### Issue: Embedding Model Not Found
**Solution**: Set explicit embedding model
```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### Issue: Slow Processing
**Solution**: Reduce max_paths_per_chunk and use fewer workers
```python
kg_extractor = SimpleLLMPathExtractor(
    llm=llm,
    max_paths_per_chunk=5,  # Reduce from 10
    num_workers=1          # Reduce from 4
)
```

### Issue: Graph Too Large to Visualize
**Solution**: Limit triplets in visualization
```python
# Show only first 50 triplets for performance
for triplet in triplets[:50]:
    # Process visualization
```

## Time-Saving Shortcuts

### Pre-built Components
```python
# Quick Streamlit layout
def create_layout():
    st.set_page_config(page_title="Knowledge Graph Builder", layout="wide")
    col1, col2 = st.columns([1, 2])
    return col1, col2

# Quick graph stats
def show_stats(G):
    st.metric("Nodes", G.number_of_nodes())
    st.metric("Edges", G.number_of_edges())
    st.metric("Components", nx.number_connected_components(G.to_undirected()))
```

### Sample Test Data
```python
SAMPLE_TEXT = """
Apple Inc. is an American multinational technology company headquartered in Cupertino, California. 
It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. 
The company is known for products like the iPhone, iPad, and Mac computers.
Tim Cook became CEO in 2011 after Steve Jobs resigned.
"""

SAMPLE_ENTITIES = [
    {"name": "Apple Inc.", "type": "ORGANIZATION", "description": "Technology company"},
    {"name": "Steve Jobs", "type": "PERSON", "description": "Co-founder of Apple"},
    {"name": "iPhone", "type": "CONCEPT", "description": "Smartphone product"}
]
```

## Debugging Tips

### Enable Debug Mode
```python
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

def debug_print(msg):
    if DEBUG:
        st.sidebar.write(f"🔍 {msg}")
```

### Common Debug Points
1. After document loading - check chunk count
2. After extraction - verify JSON structure
3. After graph building - check node/edge count
4. Before visualization - verify graph connectivity

## Final Checklist
- [ ] API key in .env file
- [ ] All imports at top of files
- [ ] Error handling for all external calls
- [ ] Progress indicators for long operations
- [ ] Sample data for quick testing
- [ ] Clean session state management
- [ ] Graph size limits enforced
- [ ] Proper JSON validation

## Emergency Fallbacks

If running out of time:
1. Hardcode sample graph data
2. Use regex for basic entity extraction
3. Skip URL processing
4. Simplify to single-file processing
5. Pre-compute Q&A responses for demo

## Useful Code Snippets

### Quick Entity Extraction without LLM
```python
import re
def quick_extract_entities(text):
    # Extract capitalized words as potential entities
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    return [{"name": e, "type": "ENTITY", "description": ""} for e in set(entities)]
```

### Simple Graph Visualization Fallback
```python
import matplotlib.pyplot as plt
def quick_visualize(G):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=8, arrows=True)
    st.pyplot(plt)
```

Remember: **Focus on working demo over perfect code!**