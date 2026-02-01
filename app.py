import streamlit as st
from sentence_transformers import SentenceTransformer
from supabase import create_client
import os
from dotenv import load_dotenv

# 1. SETUP
load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

# 2. LOAD AI MODEL (Cached)
# We use @st.cache_resource so the app doesn't reload the 
# heavy AI model every time you type a letter. It loads once and stays ready.
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# 3. THE UI (Visuals)
st.title("ArXiv Semantic Search")
st.write("Ask a question, and I'll find the relevant math in the papers.")

query = st.text_input("Your Question:", placeholder="e.g., How does a transformer model work?")

# 4. THE SEARCH LOGIC
if query:
    st.write("Searching...")
    
    # A. Turn the question into a vector (Math)
    query_vector = model.encode(query).tolist()
    
    # B. Send vector to Supabase to find nearest neighbors
    response = supabase.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_threshold": 0.5, # Lower this if you get no results
        "match_count": 5
    }).execute()
    
    # C. Display Results
    st.markdown("### Top Matches")
    
    if not response.data:
        st.warning("No relevant matches found. Try asking a broader question.")
    else:
        for match in response.data:
            # Create a card-like view for each result
            with st.expander(f"{match['metadata']['title']} (Score: {match['similarity']:.2f})", expanded=True):
                st.write(match['content'])
                st.markdown(f"[Read PDF]({match['metadata']['url']})")