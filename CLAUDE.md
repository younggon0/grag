# CLAUDE.md - Project Guidelines for Knowledge Graph Builder

## Project Context
Building a Universal Knowledge-Graph Builder for a 2-hour hackathon using Anthropic Claude API with LangChain.

## API Configuration

### Required Environment Variables
```bash
ANTHROPIC_API_KEY=your_api_key_here
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
import json
import os
from typing import List, Dict, Any

# Third-party
import streamlit as st
import networkx as nx
from pyvis.network import Network
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Local
from components import graph_extractor
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
if 'graph' not in st.session_state:
    st.session_state.graph = nx.DiGraph()
    st.session_state.processed_docs = []
    st.session_state.extraction_cache = {}
```

## Performance Guidelines

### Caching Strategy
1. Cache LLM responses by text hash
2. Store processed documents in session state
3. Reuse graph object across reruns

### Batch Processing
```python
# Process chunks in batches of 5
BATCH_SIZE = 5
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i+BATCH_SIZE]
    # Process batch in parallel if possible
```

### Graph Size Limits
- Max 500 nodes for visualization
- Max 1000 edges
- Implement pagination for larger graphs

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

### Issue: Slow Processing
**Solution**: Reduce chunk size, use Haiku model, implement caching

### Issue: Malformed JSON from LLM
**Solution**: 
```python
import re
def extract_json(text):
    # Find JSON block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return {"entities": [], "relationships": []}
```

### Issue: Graph Too Large to Visualize
**Solution**: Filter by centrality
```python
# Show only top 100 nodes by degree
top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:100]
subgraph = G.subgraph([n for n, d in top_nodes])
```

### Issue: Duplicate Entities
**Solution**: Simple deduplication
```python
def dedupe_entities(entities):
    seen = {}
    for e in entities:
        key = e['name'].lower().strip()
        if key not in seen or len(e.get('description', '')) > len(seen[key].get('description', '')):
            seen[key] = e
    return list(seen.values())
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