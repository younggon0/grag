# URL Ingestion Feature

## Overview
The Knowledge Graph Builder now supports fetching and processing content directly from web URLs. This feature extracts text content from HTML pages and converts it into a knowledge graph.

## How It Works

### 1. URL Input
- Select "Enter URL" from the input method options in the sidebar
- Enter any valid HTTP or HTTPS URL
- Click "Fetch URL Content" to retrieve the page

### 2. HTML Processing
The system intelligently extracts content from HTML pages:
- **Title Extraction**: Captures page title for context
- **Main Content Detection**: Identifies article, main, or content divs
- **Structured Text**: Preserves headings (H1-H4) hierarchy
- **Clean Extraction**: Removes scripts, styles, navigation, and ads
- **Smart Filtering**: Excludes short text fragments (likely navigation/ads)

### 3. Metadata Tracking
URLs are processed with enhanced metadata:
- Source URL and domain
- Fetch timestamp
- Source type identification
- Automatic attribution in Q&A responses

## Supported Content

### ✅ Works Well With:
- News articles
- Wikipedia pages
- Blog posts
- Documentation sites
- Static HTML content
- Text-heavy pages

### ⚠️ Limitations:
- JavaScript-rendered content may not be fully captured
- PDF files are not supported (HTML only)
- Very large pages (>100MB) are rejected
- Authentication-required pages won't work
- Some sites may block automated requests

## Usage Examples

### Example 1: Wikipedia Article
```
URL: https://en.wikipedia.org/wiki/Artificial_intelligence
Result: Extracts main article content, preserving section structure
```

### Example 2: News Article
```
URL: https://www.example-news.com/article
Result: Captures headline, body text, and quotes
```

### Example 3: Technical Documentation
```
URL: https://docs.example.com/guide
Result: Preserves code examples, headings, and lists
```

## Testing

Run the test scripts to verify URL functionality:

```bash
# Test basic URL fetching
python test_url_fetch.py

# Test with specific URL
python test_url_fetch.py "https://example.com/article"

# Test full pipeline (URL to knowledge graph)
python test_url_integration.py
```

## Technical Details

### HTML Parsing Strategy
1. Fetch with browser-like User-Agent headers
2. Parse with BeautifulSoup (lxml parser)
3. Remove non-content elements
4. Identify main content area
5. Extract structured text
6. Clean and normalize output

### Content Size Limits
- Minimum: 100 characters (to ensure meaningful content)
- Maximum: 100MB (to prevent memory issues)

### Error Handling
- Invalid URLs are rejected with user-friendly message
- Network timeouts after 30 seconds
- Parsing failures fall back to basic text extraction
- All errors are logged for debugging

## Future Enhancements
- [ ] Support for PDF documents
- [ ] JavaScript rendering with headless browser
- [ ] Batch URL processing
- [ ] Sitemap crawling
- [ ] RSS feed integration
- [ ] Authentication support
- [ ] Content caching