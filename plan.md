# Universal Knowledge-Graph Builder - Implementation Plan (v2)

## Project Overview
**Goal**: Build a Streamlit app that converts documents into an interactive knowledge graph with NL Q&A capabilities.
**Time Limit**: 2 hours (original) + 1 hour enhancement
**Key Requirements**:
- Ingest TXT files and URLs (≤ 100 MB total)
- Build a graph of concepts with node/edge visualization
- Support natural language questions over the graph
- **NEW**: Maintain document indexing with source attribution
- **NEW**: Link entities back to original document chunks

## Technical Stack

### Current Implementation
- **LLM**: Anthropic Claude API (via LangChain)
- **Framework**: LangChain for orchestration
- **UI**: Streamlit
- **Graph Processing**: NetworkX
- **Graph Database**: Neo4j for persistence
- **Visualization**: Pyvis (interactive HTML graphs)
- **Document Processing**: LangChain text splitters

### Proposed Enhancement with LlamaIndex
- **Graph Framework**: LlamaIndex PropertyGraphIndex
- **Graph Store**: Neo4jPropertyGraphStore (built-in integration)
- **Extraction**: SimpleLLMPathExtractor / SchemaLLMPathExtractor
- **Query Engine**: GraphRAGQueryEngine with source attribution
- **Document Index**: Automatic chunk-to-entity linking

## Implementation Timeline (2 Hours)

### Phase 1: Setup & Foundation (20 mins)
#### 1.1 Project Setup (5 mins)
- Create virtual environment
- Install dependencies
- Set up API keys in `.env`

#### 1.2 Core Structure (15 mins)
- Create main `app.py`
- Set up modular components
- Basic Streamlit layout

### Phase 2: Document Processing (25 mins)
#### 2.1 Document Ingestion (15 mins)
- Streamlit file uploader for TXT files
- URL input with web scraping
- LangChain document loaders:
  - `TextLoader` for TXT files
  - `WebBaseLoader` for URLs

#### 2.2 Text Processing (10 mins)
- Use `RecursiveCharacterTextSplitter`
- Chunk size: 1000 chars with 200 overlap
- Metadata preservation

### Phase 3: Knowledge Graph Extraction (40 mins)
#### 3.1 Entity & Relationship Extraction (25 mins)
```python
# Core extraction prompt template
extraction_prompt = """
Extract entities and relationships from this text.
Return JSON with:
{
  "entities": [
    {"name": "...", "type": "...", "description": "..."}
  ],
  "relationships": [
    {"source": "...", "target": "...", "type": "...", "description": "..."}
  ]
}

Text: {text}
"""
```
- Use Claude 3 Haiku for speed
- Batch process chunks
- JSON output parsing with retry logic

#### 3.2 Graph Construction (15 mins)
- NetworkX DiGraph creation
- Entity deduplication by name similarity
- Relationship aggregation
- Node/edge attributes storage

### Phase 4: Visualization (20 mins)
#### 4.1 Pyvis Graph Generation (15 mins)
- Color coding by entity type
- Interactive physics engine
- Node size by importance (degree centrality)
- Edge labels for relationship types

#### 4.2 Streamlit Integration (5 mins)
- HTML component embedding
- Graph statistics sidebar
- Download graph as JSON

### Phase 5: Q&A System (10 mins)
#### 5.1 Context Retrieval
- Find relevant nodes by keyword matching
- Extract k-hop neighborhood
- Format subgraph as context

#### 5.2 Answer Generation
```python
qa_prompt = """
Given this knowledge graph context:
{graph_context}

Answer this question: {question}

Provide a clear answer with references to entities and relationships.
"""
```

### Phase 6: Polish & Testing (5 mins)
- Error handling
- Loading states
- Sample documents
- Quick testing

## Enhancement Plan - LlamaIndex Integration (1 Hour)

### Phase 1: Research & Planning (10 mins)
- Review LlamaIndex GraphRAG v2 documentation
- Identify gap between current implementation and LlamaIndex approach
- Plan migration strategy

### Phase 2: LlamaIndex Setup (15 mins)
```bash
pip install llama-index-core
pip install llama-index-llms-anthropic
pip install llama-index-graph-stores-neo4j
pip install llama-index-embeddings-openai  # or another embedder
```

### Phase 3: Refactor Core Components (20 mins)
```python
from llama_index.core import PropertyGraphIndex, Document
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.anthropic import Anthropic
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

# Initialize Neo4j store
graph_store = Neo4jPropertyGraphStore(
    uri=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# Create extractor with Claude
llm = Anthropic(api_key=ANTHROPIC_API_KEY, model="claude-3-haiku-20240307")
kg_extractor = SimpleLLMPathExtractor(llm=llm)

# Build index with document tracking
index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    kg_extractors=[kg_extractor],
    show_progress=True
)
```

### Phase 4: Update Q&A with Source Attribution (10 mins)
```python
# Query with source tracking
query_engine = index.as_query_engine(
    include_text=True,  # Include source text
    response_mode="tree_summarize",
    verbose=True
)

response = query_engine.query("Who founded Apple?")
# response.source_nodes contains the original chunks
```

### Phase 5: UI Updates (5 mins)
- Add source display in Q&A responses
- Show which document/chunk entities came from
- Add "View Source" buttons for entities

## Component Architecture

```
grag/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Dependencies
├── .env                      # API keys
├── components/
│   ├── __init__.py
│   ├── document_processor.py # Document ingestion & chunking
│   ├── graph_extractor.py    # Entity/relationship extraction
│   ├── graph_builder.py      # NetworkX graph construction
│   ├── visualizer.py         # Pyvis visualization
│   └── qa_engine.py          # Q&A functionality
├── utils/
│   ├── __init__.py
│   └── prompts.py            # Prompt templates
└── sample_docs/              # Test documents
```

## Key Implementation Details

### Current vs LlamaIndex Approach

| Feature | Current Implementation | LlamaIndex Enhancement |
|---------|----------------------|----------------------|
| Entity Extraction | Custom Claude prompts | SimpleLLMPathExtractor |
| Document Index | Not implemented | PropertyGraphIndex (automatic) |
| Source Attribution | Not available | Built-in with source_nodes |
| Neo4j Integration | Custom manager | Neo4jPropertyGraphStore |
| Chunk-Entity Links | Manual tracking needed | Automatic |
| Query Engine | Custom context retrieval | GraphRAGQueryEngine |
| Graph Communities | Not implemented | Built-in clustering |

### Benefits of LlamaIndex Migration
1. **Document Tracking**: Automatic linking of entities to source chunks
2. **Source Citations**: Every answer includes source references
3. **Better Retrieval**: Hybrid search (graph + vector + keyword)
4. **Production Ready**: Battle-tested components
5. **Less Code**: ~50% reduction in custom code
6. **Maintenance**: Community-maintained and updated

### Entity Extraction Strategy
1. Process each chunk independently
2. Extract 5-10 entities per chunk max
3. Focus on proper nouns, concepts, and key terms
4. Use consistent entity types (PERSON, ORGANIZATION, CONCEPT, LOCATION, EVENT)

### Relationship Extraction
1. Limit to entities within same chunk
2. Common relationship types: "relates_to", "part_of", "causes", "created_by"
3. Include confidence scores if time permits

### Performance Optimizations
1. Use Claude 3 Haiku for extraction (faster, cheaper)
2. Cache processed documents in session state
3. Limit graph to 500 nodes for visualization
4. Progressive loading for large documents

### Error Handling
1. Graceful fallback for failed extractions
2. Chunk retry with exponential backoff
3. User feedback for processing status
4. Validation for file size limits

## Success Metrics

### Original Goals (Completed ✅)
- [x] Successfully load and process TXT files
- [x] Extract and visualize at least 20 entities
- [x] Display interactive graph with zoom/pan
- [x] Answer at least 3 different types of questions
- [x] Complete implementation within 2 hours

### Enhanced Goals (With LlamaIndex)
- [ ] Implement document indexing with PropertyGraphIndex
- [ ] Link entities to source document chunks
- [ ] Show source attribution in Q&A responses
- [ ] Enable chunk-level retrieval
- [ ] Add graph community detection
- [ ] Display document sources in UI
- [ ] Test cross-document relationships with attribution

## Contingency Plans

### If Running Out of Time:
1. **Skip URL processing** - Focus on TXT files only
2. **Simplify extraction** - Use regex for basic entity detection
3. **Basic visualization** - Use matplotlib instead of Pyvis
4. **Hardcode Q&A** - Pre-compute answers for demo questions

### If Ahead of Schedule:
1. Add entity resolution/merging
2. Implement graph filtering by entity type
3. Add export to Neo4j format
4. Include confidence scores
5. Multi-document graph merging

## Testing Strategy
1. Start with small text (1-2 paragraphs)
2. Test with Wikipedia article about a person
3. Verify graph has connected components
4. Test Q&A with "Who", "What", "How" questions

## Demo Script
1. Upload sample document about "Apple Inc."
2. Show extracted entities (Steve Jobs, products, dates)
3. Demonstrate graph interaction
4. Ask: "Who founded Apple?"
5. Ask: "What products are mentioned?"
6. Show graph statistics

## Critical Success Factors
1. **Keep It Simple**: Focus on core functionality
2. **Fail Fast**: Test each component immediately
3. **User Feedback**: Show progress indicators
4. **Demo Ready**: Have fallback data if live processing fails
5. **Time Management**: Strict adherence to timeline

## Resources & References
- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)
- [Anthropic Claude API](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [Pyvis Documentation](https://pyvis.readthedocs.io/)
- [NetworkX Tutorial](https://networkx.org/documentation/stable/tutorial.html)