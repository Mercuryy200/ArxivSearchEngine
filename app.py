import streamlit as st
from sentence_transformers import SentenceTransformer
from supabase import create_client
import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. SETUP: Load keys
load_dotenv()
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
google_key = os.environ.get("GOOGLE_API_KEY")

# Initialize Clients
supabase = create_client(supabase_url, supabase_key)
genai.configure(api_key=google_key)

# 2. CACHING (Load models once)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# 3. UI SETUP
st.set_page_config(page_title="ArXiv Chat", layout="wide")
st.title("🤖 ArXiv Chatbot")
st.markdown("""
Ask a question, and I will:
1. **Search** for relevant research papers.
2. **Read** them for you.
3. **Answer** based *only* on those papers.
""")

# 4. THE CHAT INTERFACE
query = st.text_input("Ask a question about AI/Tech:", placeholder="e.g., How does Attention work in Transformers?")

if query:
    with st.spinner("Searching the database..."):
        # STEP A: RETRIEVAL (Find the relevant text)
        query_vector = embedding_model.encode(query).tolist()
        
        response = supabase.rpc("match_documents", {
            "query_embedding": query_vector,
            "match_threshold": 0.3, 
            "match_count": 5
        }).execute()
        
        matches = response.data

    if not matches:
        st.warning("No relevant papers found in the database.")
    else:
        # STEP B: PREPARE CONTEXT (Glue the text chunks together)
        # We tell the AI: "Here is the background info you need."
        context_text = "\n\n".join([f"Source ({doc['metadata']['title']}): {doc['content']}" for doc in matches])
        
        # STEP C: GENERATION (Ask Gemini)
        with st.spinner("Reading papers and generating answer..."):
            try:
                # We use the Gemini Flash model (fast & free tier friendly)
                model = genai.GenerativeModel('gemini-pro-latest')
                
                prompt = f"""
                You are a helpful research assistant. Use the following Context to answer the User's Question.
                If the answer is not in the context, say "I couldn't find that information in the papers."
                
                Context:
                {context_text}
                
                User's Question: {query}
                
                Answer:
                """
                
                ai_response = model.generate_content(prompt)
                
                # Display the Answer
                st.success("Answer generated from research papers:")
                st.markdown(f"### 💡 {ai_response.text}")
                
            except Exception as e:
                st.error(f"Error talking to AI: {e}")

        # Show sources at the bottom (Transparency)
        with st.expander("📚 View Source Documents (The 'Context')"):
            for match in matches:
                st.markdown(f"**{match['metadata']['title']}** (Similarity: {match['similarity']:.2f})")
                st.info(match['content'])
                st.markdown(f"[Link to PDF]({match['metadata']['url']})")
                st.divider()