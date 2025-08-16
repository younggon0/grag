# 🧠 Universal Knowledge Graph Builder

A Streamlit application that converts documents into interactive knowledge graphs with natural language Q&A capabilities.

## Features

- **Document Ingestion**: Upload TXT files or fetch content from URLs
- **Entity & Relationship Extraction**: Uses Claude API to intelligently extract entities and their relationships
- **Interactive Visualization**: Dynamic, physics-enabled graph visualization with Pyvis
- **Natural Language Q&A**: Ask questions about the knowledge graph and get contextual answers
- **Persistent Storage**: Automatic saving to Neo4j database (if configured)
- **Cross-Document Relationships**: Discovers connections between entities across multiple documents
- **Export Functionality**: Download your knowledge graph as JSON

## Quick Start

### Prerequisites

- Python 3.10+
- Anthropic API key
- Neo4j database (optional, for persistent storage)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd grag
```

2. Install dependencies using uv (recommended):
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

Or using pip:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up your API keys:
   - Copy `.env.example` to `.env`
   - Add your Anthropic API key
   - (Optional) Add Neo4j credentials for persistent storage:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_key
   NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   ```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Input Document**:
   - Upload a TXT file (max 100MB)
   - Enter a URL to fetch content
   - Or use the sample document

2. **Build Graph**:
   - Click "Build Knowledge Graph" to process the document
   - Watch as entities and relationships are extracted

3. **Explore**:
   - **Graph Tab**: Interactive visualization with zoom, pan, and physics simulation
   - **Q&A Tab**: Ask natural language questions about the graph
   - **Entities Tab**: View all extracted entities and relationships in tabular format

4. **Export**:
   - Download the complete knowledge graph as JSON

## Project Structure

```
grag/
├── app.py                    # Main Streamlit application
├── components/
│   ├── document_processor.py # Document ingestion & chunking
│   ├── graph_extractor.py    # Entity/relationship extraction
│   ├── graph_builder.py      # NetworkX graph construction
│   ├── visualizer.py         # Pyvis visualization
│   └── qa_engine.py          # Q&A functionality
├── sample_docs/              # Sample documents for testing
├── .env                      # API keys (not in repo)
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Technology Stack

- **LLM**: Anthropic Claude 3 Haiku
- **Framework**: LangChain for LLM orchestration
- **UI**: Streamlit
- **Graph Processing**: NetworkX
- **Graph Database**: Neo4j (optional, for persistence)
- **Visualization**: Pyvis
- **Document Processing**: BeautifulSoup4, LangChain text splitters

## Sample Questions

Once you've built a knowledge graph, try asking:
- "Who founded Apple?"
- "What companies are mentioned?"
- "What is the relationship between Steve Jobs and Apple?"
- "How many entities are in the graph?"
- "List all people mentioned"

## Persistent Storage with Neo4j

The app supports Neo4j for persistent graph storage:

- **Automatic Loading**: Graph loads from Neo4j on app startup
- **Auto-Save**: Every processed document is automatically saved to Neo4j
- **Cross-Session Persistence**: Your graph persists across browser refreshes and sessions
- **Multi-Document Relationships**: Build connections between entities across multiple documents

### Setting up Neo4j

1. **Neo4j AuraDB (Free Tier)**:
   - Sign up at [neo4j.com/aura](https://neo4j.com/aura)
   - Create a free instance
   - Copy credentials to `.env` file

2. **Local Neo4j**:
   - Download Neo4j Desktop
   - Create local database
   - Use `neo4j://localhost:7687` as URI

### Neo4j Utilities

The project includes utility scripts for managing your Neo4j database:

#### Clear Database
```bash
python clear_neo4j.py
```
Safely clears all data from Neo4j with confirmation prompt.

#### Database Management
```bash
# Show database statistics
python neo4j_utils.py stats

# Search for entities
python neo4j_utils.py search --query "Steve Jobs"

# Get entity details
python neo4j_utils.py details --query "Apple Inc."

# Export graph to JSON
python neo4j_utils.py export --output my_graph.json

# List recent entities
python neo4j_utils.py recent --limit 10

# Clear database (with confirmation)
python neo4j_utils.py clear
```

## Performance Tips

- For large documents, the app automatically chunks text for processing
- Graph visualization is limited to 500 nodes for performance
- Use the physics toggle to improve rendering of large graphs
- Enable caching by not modifying already processed documents
- Neo4j connection provides faster loading for large graphs

## Troubleshooting

**"API key not set" error**:
- Make sure you've added your Anthropic API key to the `.env` file

**Graph not displaying**:
- Try refreshing the view with the refresh button
- Toggle physics on/off
- For very large graphs, the app automatically shows top nodes by connectivity

**Slow processing**:
- Large documents are chunked and processed in batches
- First-time processing is slower; subsequent runs use cached extractions

## Deployment on Streamlit Cloud

Deploy your app for free on Streamlit Cloud in minutes:

### Prerequisites
- GitHub account
- Your code pushed to a GitHub repository
- Anthropic API key

### Step-by-Step Deployment

1. **Push your code to GitHub**:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/knowledge-graph-builder.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository and branch (main)
   - Set main file path: `app.py`

3. **Configure Secrets**:
   - Click "Advanced settings" before deploying (or go to App Settings → Secrets after deployment)
   - Add your API key in the secrets section:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-api03-YOUR-ACTUAL-KEY-HERE"
   ```

4. **Deploy**:
   - Click "Deploy"
   - Your app will be live at: `https://YOUR-APP-NAME.streamlit.app`
   - The URL will be permanent and shareable

### Files Required for Deployment

Ensure these files exist in your repository:
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version (contains: `python-3.11`)
- `.streamlit/config.toml` - Streamlit configuration
- `.env.example` - Shows required environment variables (don't commit actual `.env`)

### Deployment Tips

- **Free Tier Limits**: Streamlit Cloud free tier includes 1GB of memory and 1GB of storage
- **Private Apps**: You can make your app private in the settings
- **Custom Domain**: Available on paid plans
- **Automatic Updates**: App auto-updates when you push to GitHub
- **Logs**: View logs in the Streamlit Cloud dashboard for debugging

### Alternative Deployment Options

For other deployment options (Railway, Heroku, Google Cloud Run, etc.), consider:
- **Railway**: Simple deployment with auto-scaling (~$5-20/month)
- **Hugging Face Spaces**: Great for AI/ML demos (free)
- **Google Cloud Run**: Pay-per-use serverless (~$0-10/month)

## Development

For development guidelines and implementation details, see:
- `plan.md` - Detailed implementation plan
- `CLAUDE.md` - Project-specific guidelines and code snippets

## License

This project was created for a hackathon and is provided as-is for educational purposes.