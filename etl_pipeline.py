import os
import requests
import xml.etree.ElementTree as ET
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv

# 1. SETUP: Load keys and models
load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

# Download the model (this happens once, then it's cached)
print("Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_papers(search_query="cat:cs.AI", max_results=5):
    """
    EXTRACT: Downloads PDFs from ArXiv API
    """
    print(f"Fetching {max_results} papers for query: {search_query}...")
    api_url = f'http://export.arxiv.org/api/query?search_query={search_query}&start=0&max_results={max_results}'
    data = requests.get(api_url).content
    
    # Parse XML response
    root = ET.fromstring(data)
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
    
    # Create a 'downloads' folder if it doesn't exist
    if not os.path.exists('downloads'):
        os.makedirs('downloads')
        
    papers = []
    
    for entry in root.findall('atom:entry', namespace):
        title = entry.find('atom:title', namespace).text.replace('\n', ' ')
        link = entry.find("atom:link[@title='pdf']", namespace).attrib['href']
        published = entry.find('atom:published', namespace).text
        
        # Clean filename safely
        filename = f"downloads/{title[:20].replace(' ', '_').replace('/', '-')}.pdf"
        
        # Download the actual PDF
        if not os.path.exists(filename):
            print(f"Downloading: {title}...")
            response = requests.get(link)
            with open(filename, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Skipping download (exists): {title}")
            
        papers.append({
            "title": title,
            "path": filename,
            "url": link,
            "date": published
        })
        
    return papers

def process_and_load(papers):
    """
    TRANSFORM & LOAD: Reads text -> Chunks -> Embeds -> Saves to DB
    """
    for paper in papers:
        print(f"Processing: {paper['title']}...")
        
        # A. Read PDF Text
        try:
            reader = PdfReader(paper['path'])
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            print(f"Error reading PDF: {e}")
            continue

        # B. Chunking (Split text into 500-char pieces with overlap)
        chunk_size = 500
        overlap = 50
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 100: # Ignore tiny chunks
                chunks.append(chunk)
                
        if not chunks:
            continue

        # C. Embedding (The Math)
        print(f" - Generating vectors for {len(chunks)} chunks...")
        vectors = model.encode(chunks)
        
        # D. Load to Supabase
        data_payload = []
        for i, chunk in enumerate(chunks):
            data_payload.append({
                "content": chunk,
                "embedding": vectors[i].tolist(), # Convert numpy array to standard list
                "metadata": {
                    "title": paper['title'],
                    "url": paper['url'],
                    "published": paper['date']
                }
            })
            
        # Upload in batches of 100 to be safe
        batch_size = 100
        for i in range(0, len(data_payload), batch_size):
            batch = data_payload[i:i+batch_size]
            response = supabase.table('documents').insert(batch).execute()
            
        print(f" - Uploaded {len(chunks)} chunks to Database.")

if __name__ == "__main__":
    # 1. Get the raw files
    downloaded_papers = extract_papers(max_results=50) # Start small with 3 papers
    
    # 2. Process and Upload
    process_and_load(downloaded_papers)
    
    print("Pipeline Finished Successfully!")