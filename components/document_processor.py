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
        Fetch content from a URL
        
        Args:
            url: URL to fetch
            
        Returns:
            Text content from the URL
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
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