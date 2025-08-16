"""
Document Processor Module
Handles document ingestion from files and URLs
"""

import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def fetch_url(self, url: str) -> str:
        """
        Fetch content from a URL with enhanced HTML extraction
        
        Args:
            url: URL to fetch
            
        Returns:
            Text content from the URL with preserved structure
        """
        try:
            # Set headers to appear as a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove non-content elements
            for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'header', 'meta', 'link', 'noscript']):
                element.decompose()
            
            # Build structured text with better context preservation
            text_parts = []
            
            # Extract title
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
                if title:
                    text_parts.append(f"Title: {title}\n")
            
            # Try to find main content area (common patterns)
            main_content = (
                soup.find('main') or 
                soup.find('article') or 
                soup.find('div', {'class': ['content', 'main-content', 'post-content', 'entry-content']}) or
                soup.find('div', {'id': ['content', 'main-content', 'main']}) or
                soup.body
            )
            
            if main_content:
                # Extract headings with hierarchy
                for heading in main_content.find_all(['h1', 'h2', 'h3', 'h4']):
                    heading_text = heading.get_text().strip()
                    if heading_text:
                        # Add spacing based on heading level for structure
                        level = int(heading.name[1])
                        text_parts.append('\n' * (4 - level) + heading_text + '\n')
                
                # Extract paragraphs and lists
                for element in main_content.find_all(['p', 'li']):
                    element_text = element.get_text().strip()
                    # Filter out very short paragraphs (likely navigation or ads)
                    if len(element_text) > 30:
                        text_parts.append(element_text)
                
                # Extract blockquotes
                for quote in main_content.find_all('blockquote'):
                    quote_text = quote.get_text().strip()
                    if quote_text:
                        text_parts.append(f"\nQuote: {quote_text}\n")
            
            # Fallback to basic text extraction if no main content found
            if not text_parts:
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text_parts = [chunk for chunk in chunks if chunk and len(chunk) > 30]
            
            # Join with proper spacing
            final_text = '\n\n'.join(text_parts)
            
            # Clean up excessive whitespace
            import re
            final_text = re.sub(r'\n{3,}', '\n\n', final_text)
            final_text = re.sub(r' {2,}', ' ', final_text)
            
            return final_text.strip()
            
        except requests.RequestException as e:
            raise Exception(f"Error fetching URL: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing URL content: {str(e)}")
    
    def process_text(self, text: str) -> List[str]:
        """
        Process text into chunks
        
        Args:
            text: Text to process
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Filter out empty chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        return chunks
    
    def process_file(self, file_path: str) -> List[str]:
        """
        Process a text file
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of text chunks
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return self.process_text(text)
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")