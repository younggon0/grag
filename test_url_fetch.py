#!/usr/bin/env python3
"""
Test script for URL fetching functionality
"""

from components.document_processor import DocumentProcessor
import sys

def test_url_fetch(url):
    """Test fetching and processing a URL"""
    processor = DocumentProcessor()
    
    print(f"\n🌐 Testing URL: {url}")
    print("=" * 60)
    
    try:
        # Fetch content
        content = processor.fetch_url(url)
        
        # Display statistics
        print(f"✅ Successfully fetched content")
        print(f"📊 Content length: {len(content):,} characters")
        print(f"📝 Number of lines: {content.count(chr(10)):,}")
        
        # Show preview
        print("\n📄 Content Preview (first 500 chars):")
        print("-" * 40)
        preview = content[:500] + "..." if len(content) > 500 else content
        print(preview)
        print("-" * 40)
        
        # Check for title
        if "Title:" in content[:100]:
            title_end = content.find('\n', content.find("Title:"))
            if title_end > 0:
                title = content[content.find("Title:"):title_end]
                print(f"\n📌 {title}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    # Test URLs
    test_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://www.anthropic.com/claude",
        "https://example.com"
    ]
    
    if len(sys.argv) > 1:
        # Test user-provided URL
        test_urls = [sys.argv[1]]
    
    success_count = 0
    for url in test_urls:
        if test_url_fetch(url):
            success_count += 1
    
    print(f"\n\n✨ Testing complete: {success_count}/{len(test_urls)} URLs fetched successfully")