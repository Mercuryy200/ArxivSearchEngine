import os
import re
import time
import requests
import xml.etree.ElementTree as ET
import nltk
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv

# Download sentence tokenizer data on first run (no-op if already present).
nltk.download('punkt_tab', quiet=True)

_ATOM_NS  = {'atom': 'http://www.w3.org/2005/Atom'}
_ARXIV_NS = 'http://arxiv.org/schemas/atom'

# 1. SETUP: Load keys and models
load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

# Download the model (this happens once, then it's cached)
print("Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def _get_with_retry(url, timeout=90, max_retries=3):
    """GET with exponential backoff on timeout or server errors."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout):
            wait = 2 ** attempt * 5  # 5s, 10s, 20s
            print(f"Request timed out (attempt {attempt + 1}/{max_retries}), retrying in {wait}s...")
            if attempt < max_retries - 1:
                time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 503 and attempt < max_retries - 1:
                wait = 2 ** attempt * 5
                print(f"ArXiv returned 503 (attempt {attempt + 1}/{max_retries}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise requests.exceptions.RetryError(f"All {max_retries} attempts failed for {url}")


def extract_papers(
    search_query: str = (
        "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV"
        " OR cat:q-fin.ST OR cat:q-fin.CP OR cat:q-fin.PM"
        " OR cat:q-fin.TR OR cat:q-fin.RM OR cat:q-fin.MF"
    ),
    max_results: int = 5,
) -> list:
    """
    EXTRACT: Downloads PDFs from ArXiv API.
    Parses title, URL, published date, and primary category for each entry.
    File-level deduplication: skips download if local PDF already exists.
    """
    print(f"Fetching {max_results} papers for query: {search_query}...")
    api_url = (
        f'http://export.arxiv.org/api/query'
        f'?search_query={search_query}'
        f'&start=0'
        f'&max_results={max_results}'
        f'&sortBy=submittedDate'
        f'&sortOrder=descending'
    )
    data = _get_with_retry(api_url)

    # Parse XML response
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        print("Warning: ArXiv returned malformed XML — retrying with smaller batch...")
        fallback_url = api_url.replace(f'max_results={max_results}', 'max_results=50')
        data = _get_with_retry(fallback_url)
        root = ET.fromstring(data)

    # Create a 'downloads' folder if it doesn't exist
    if not os.path.exists('downloads'):
        os.makedirs('downloads')

    papers = []

    for entry in root.findall('atom:entry', _ATOM_NS):
        title_el = entry.find('atom:title', _ATOM_NS)
        if title_el is None:
            continue
        title = title_el.text.replace('\n', ' ').strip()

        link_el = entry.find("atom:link[@title='pdf']", _ATOM_NS)
        if link_el is None:
            continue
        link = link_el.attrib['href']

        published_el = entry.find('atom:published', _ATOM_NS)
        published = published_el.text if published_el is not None else ''

        # Parse category — prefer arxiv:primary_category, fall back to atom:category
        primary_cat_el = entry.find(f'{{{_ARXIV_NS}}}primary_category')
        if primary_cat_el is not None:
            category = primary_cat_el.attrib.get('term', '')
        else:
            cat_el = entry.find('atom:category', _ATOM_NS)
            category = cat_el.attrib.get('term', '') if cat_el is not None else ''

        # Clean filename safely
        filename = f"downloads/{title[:20].replace(' ', '_').replace('/', '-')}.pdf"

        # Download the actual PDF (file-level deduplication)
        if not os.path.exists(filename):
            print(f"Downloading: {title}...")
            pdf_data = _get_with_retry(link, timeout=120)
            with open(filename, 'wb') as f:
                f.write(pdf_data)
        else:
            print(f"Skipping download (exists): {title}")

        papers.append({
            "title":    title,
            "path":     filename,
            "url":      link,
            "date":     published,
            "category": category,
        })

    return papers

def process_and_load(papers):
    """
    TRANSFORM & LOAD: Reads text -> Chunks -> Embeds -> Saves to DB
    Skips papers whose URL is already present in the documents table so that
    re-running the pipeline never creates duplicate chunks.
    """
    for paper in papers:
        # Check whether this paper's chunks already exist in the database.
        # We filter on metadata->>'url' (PostgREST JSONB path syntax).
        existing = (
            supabase.table('documents')
            .select('id')
            .filter('metadata->>url', 'eq', paper['url'])
            .limit(1)
            .execute()
        )
        if existing.data:
            print(f"Skipping (already indexed): {paper['title']}")
            continue

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

        # Sanitise extracted text before chunking.
        #
        # 1. Remove ASCII control characters (PostgreSQL text type rejects them
        #    and they add no semantic value).  Keep tab \x09, newline \x0a,
        #    carriage-return \x0d which are normal whitespace in PDFs.
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        #
        # 2. Drop lone Unicode surrogates (\ud800–\udfff).  pypdf occasionally
        #    produces these from broken PDF font tables.  They are legal Python
        #    str values but invalid UTF-8, so the Rust tokenizer crashes on
        #    them.  Round-tripping through UTF-8 with errors='ignore' removes
        #    them silently.
        text = text.encode('utf-8', errors='ignore').decode('utf-8')

        # B. Chunking — sentence-boundary chunks (target ≈ 800 chars, min 100 chars).
        # Sentences are accumulated until the next sentence would exceed the target;
        # the current buffer is then flushed as a chunk.  This keeps full sentences
        # intact so the embedding model receives coherent semantic units rather than
        # fragments mid-sentence.
        TARGET_CHUNK = 800
        MIN_CHUNK    = 100
        sentences    = nltk.sent_tokenize(text)
        chunks: list[str] = []
        current = ""
        for sent in sentences:
            if not current:
                current = sent
            elif len(current) + 1 + len(sent) <= TARGET_CHUNK:
                current += " " + sent
            else:
                if len(current) >= MIN_CHUNK:
                    chunks.append(current)
                current = sent
        if current and len(current) >= MIN_CHUNK:
            chunks.append(current)

        if not chunks:
            continue

        # Guard: drop any chunk that is not a plain non-empty string so the
        # tokenizer never receives an unexpected type or empty input.
        chunks = [c for c in chunks if isinstance(c, str) and c.strip()]
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
                    "title":     paper['title'],
                    "url":       paper['url'],
                    "published": paper['date'],
                    "category":  paper['category'],
                }
            })
            
        # Upload in batches of 100 to be safe
        batch_size = 100
        for i in range(0, len(data_payload), batch_size):
            batch = data_payload[i:i+batch_size]
            # Serialize embedding as a plain list so jsonb cast works
            rpc_batch = [
                {
                    "content":   row["content"],
                    "embedding": row["embedding"],
                    "metadata":  row["metadata"],
                }
                for row in batch
            ]
            supabase.rpc("insert_documents", {"rows": rpc_batch}).execute()
            
        print(f" - Uploaded {len(chunks)} chunks to Database.")

if __name__ == "__main__":
    # 1. Get the raw files
    downloaded_papers = extract_papers(
        search_query=(
            "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV"
            " OR cat:q-fin.ST OR cat:q-fin.CP OR cat:q-fin.PM"
            " OR cat:q-fin.TR OR cat:q-fin.RM OR cat:q-fin.MF"
        ),
        max_results=100,
    )
    
    # 2. Process and Upload
    process_and_load(downloaded_papers)
    
    print("Pipeline Finished Successfully!")
