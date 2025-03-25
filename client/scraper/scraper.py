import asyncio
import aiohttp
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from typing import List
import requests

async def fetch_url(session, url: str) -> str:
    """Asynchronously fetch content from a URL."""
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def extract_text(html: str) -> str:
    """Extract readable text from HTML content."""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text
    text = soup.get_text()
    
    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    
    # Drop blank lines
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text

def split_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks of approximately equal size."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_size += len(word) + 1  # +1 for space
        if current_size > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

async def process_urls(urls: List[str]) -> List[Document]:
    """Process multiple URLs asynchronously and return documents."""
    async with aiohttp.ClientSession() as session:
        # Fetch all URLs asynchronously
        html_contents = await asyncio.gather(
            *[fetch_url(session, url) for url in urls]
        )
        
        documents = []
        for url, html in zip(urls, html_contents):
            if html:  # Only process if we got content
                text = extract_text(html)
                chunks = split_text(text)
                
                # Create Document objects for each chunk
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={"source": url}
                    )
                    documents.append(doc)
        
        return documents
