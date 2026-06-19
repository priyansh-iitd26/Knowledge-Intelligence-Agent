import streamlit as st
import tempfile, os, json, re
from workflow import workflow, persist_dir, embeddings, llm
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
if not OPENAI_API_KEY:
    st.error("Please use your OpenAI API key to run this app.")
    st.stop()

st.set_page_config(page_title="Multi-Source Knowledge Intelligence Agent", layout="wide")
st.title("📚 Multi-Source Knowledge Intelligence Agent")

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def detect_source_type(url: str = None, file_path: str = None) -> str:
    """Detect and return the appropriate source type based on URL or file extension."""
    if url:
        url = url.lower()
        if "youtube.com" in url or "youtu.be" in url:
            return "youtube"
        elif "arxiv.org" in url:
            return "arxiv"
        elif "notion.so" in url:
            return "notion"
        elif url.endswith(".html") or url.endswith(".htm"):
            return "html"
        elif url.endswith(".pdf"):
            return "pdf"
        elif url.endswith(".docx"):
            return "docx"
        elif url.endswith(".pptx"):
            return "pptx"
        elif url.endswith(".csv"):
            return "csv"
        elif url.endswith(".json"):
            return "json"
        elif url.endswith(".txt"):
            return "txt"
        else:
            return "web"
    elif file_path:
        ext = os.path.splitext(file_path)[-1].lower()
        ext_map = {
            ".pdf": "pdf",
            ".docx": "docx",
            ".pptx": "pptx",
            ".csv": "csv",
            ".json": "json",
            ".txt": "txt",
            ".html": "html",
            ".htm": "html"
        }
        return ext_map.get(ext, "web")
    else:
        return "web"
    
intent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, api_key=OPENAI_API_KEY)

def classify_query_intent(query: str) -> str:
    """
    Classify query type: 'personal', 'knowledge', or 'hybrid'.
    """
    prompt = f"""
    Classify the following user query into one of these categories:
    1. personal → asking about the user's own details or past interactions
    2. knowledge → asking about factual or research-related information
    3. hybrid → combines personal and factual context
    
    Query: "{query}"
    Respond with just one of: "personal", "knowledge", or "hybrid".
    """
    try:
        response = intent_llm.invoke(prompt)
        return response.content.strip().lower()
    except Exception:
        return "knowledge"  # default fallback
    
def recall_relevant_facts(user_query: str) -> list:
    """ Recall personal facts previously stored in memory """
    if "chat_history" not in st.session_state:
        return []
    
    personal_facts = []
    history = st.session_state["chat_history"]
    
    patterns = {
        "name": r"\b(?:my name is|i'm called|call me|you can call me|this is)\s+([A-Za-z][A-Za-z\s'-]+)",
        "location": r"\b(?:i (?:live|reside|stay) in|i'm from|i was born in|currently in)\s+([A-Za-z\s,]+)",
        "education": r"\b(?:i (?:study|studied|am studying|majored in|graduated in|have a degree in))\s+([A-Za-z\s,&-]+)",
        "job": r"\b(?:i (?:work as|am|am a|am an|serve as|have been working as))\s+([A-Za-z\s,-]+)",
        "interest": r"\b(?:i (?:like|love|enjoy|am into|am passionate about|have an interest in))\s+([A-Za-z\s,&-]+)",
        "age": r"\b(?:i am|i'm)\s+(\d{1,3})\s*(?:years old|yo|yrs old)?",
        "email": r"\b(?:my email is|reach me at|contact me at)\s+([\w\.-]+@[\w\.-]+\.\w+)",
        "phone": r"\b(?:my phone number is|you can call me at|reach me on)\s*(\+?\d[\d\s-]{7,})",
        "experience": r"\b(?:i have|i've had|my experience includes)\s+(?:about|over|around)?\s*(\d+\+?\s*(?:years|months))",
        "language": r"\b(?:i speak|i can speak|i know|i'm fluent in)\s+([A-Za-z\s,]+)",
        "skills": r"\b(?:i (?:have skills in|am skilled at|am good at|specialize in))\s+([A-Za-z\s,&-]+)"
    }
    
    # Scan through all previous user messages
    for msg in history:
        if msg["role"] == "user":
            text = msg["content"].lower()
            for key, pattern in patterns.items():
                match = re.search(pattern, text)
                if match:
                    fact = f"{key.capitalize()}: {match.group(1).strip().capitalize()}"
                    personal_facts.append(fact)

    # If nothing matched, return the last few messages as general context
    if not personal_facts:
        context_window = [msg["content"] for msg in history[-4:] if msg["role"] == "user"]
        return context_window

    # Rank relevance to current query
    query_words = set(user_query.lower().split())
    scored_facts = sorted(
        personal_facts,
        key=lambda fact: len(set(fact.lower().split()) & query_words),
        reverse=True
    )

    return scored_facts[:3]

def clear_previous_results():
    keys_to_clear = [
        "workflow_result",
        "chat_history",
        "chat_chain",
        "chat_memory",
        "uploaded_file_path",
        "url_input"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.cache_data.clear()
    st.cache_resource.clear()
            
# Initialize session state
if "input_mode" not in st.session_state:
    st.session_state["input_mode"] = None
    
# Mode selection
col1, col2 = st.columns(2)

def reset_state():
    """Completely clears session state except specified keys."""
    for key in ["workflow_result", "chat_history", "chat_chain", "chat_memory"]:
        if key in st.session_state:
            del st.session_state[key]
    st.cache_data.clear()
    st.cache_resource.clear()

# Mode buttons
with col1:
    if st.button("🌐 Use URL"):
        if st.session_state.get("input_mode") != "url":
            reset_state()
            st.session_state["input_mode"] = "url"
            st.rerun()

with col2:
    if st.button("📁 Upload File"):
        if st.session_state.get("input_mode") != "file":
            reset_state()
            st.session_state["input_mode"] = "file"
            st.rerun()

# Input Widgets
if st.session_state.get("input_mode") == "url":
    st.subheader("🌐 Enter a URL:")
    url_input = st.text_input(
        "URL:",
        placeholder="https://example.com",
        key="url_input_key"
    )
    
    if st.button("🚀 Run URL Analysis", key="run_url"):
        # Clear old state
        for key in ["workflow_result", "chat_chain", "chat_history", "chat_memory"]:
            if key in st.session_state:
                del st.session_state[key]
        
        if not url_input:
            st.warning("Please enter a valid URL.")
        else:
            st.info("⏳ Processing URL... please wait.")
            with st.spinner("Running workflow..."):
                try:
                    result = workflow.invoke(
                        {"url": url_input.strip(),
                         "source_type": detect_source_type(url=url_input.strip())},
                        config={"configurable": {"thread_id": "knowledge_intelligence_workflow_url" }}
                    )
                    st.session_state["workflow_result"] = result
                    st.success("✅ URL successfully processed and analyzed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to process URL: {e}")
        
elif st.session_state.get("input_mode") == "file":
    st.subheader("📁 Upload a File:")
    uploaded_file = st.file_uploader(
        "Upload file",
        type=["pdf","docx","pptx","csv","json","txt"],
        key="file_uploader_key"
    ) 
    
    if uploaded_file:
        suffix = os.path.splitext(uploaded_file.name)[-1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(uploaded_file.getbuffer())
        temp_file.close()
        uploaded_file_path = temp_file.name
        
        if st.button("🚀 Run File Analysis", key="run_file"):
            for key in ["workflow_result", "chat_chain", "chat_history", "chat_memory"]:
                if key in st.session_state:
                    del st.session_state[key]
                    
            st.info("⏳ Processing file... please wait.")
            with st.spinner("Running workflow..."):
                try:
                    result = workflow.invoke(
                        {"file_path": uploaded_file_path,
                         "source_type": detect_source_type(file_path=uploaded_file_path)},
                        config={"configurable": {"thread_id": "knowledge_intelligence_workflow_file" }}
                    )
                    st.session_state["workflow_result"] = result
                    st.success("✅ File successfully processed and analyzed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to process uploadedfile: {e}")
    else:
        st.info("Please upload a file to begin.")
        
# Stop displaying results if nothing is selected
if st.session_state.get("input_mode") not in ["url", "file"]:
    st.stop()
    
# Clear results if a new mode is selected
if "workflow_result" in st.session_state:
    if st.session_state.get("input_mode") not in ["url", "file"]:
        del st.session_state["workflow_result"]

if not st.session_state.get("workflow_result"):
    st.stop()
             
# Display the results
if st.session_state.get("workflow_result") and st.session_state.get("input_mode") in ["url", "file"]:
    result = st.session_state["workflow_result"]
    
    # Final Report
    st.header("🧾 Final Intelligence Report")
    
    # Summary
    if "summary" in result:
        st.subheader("🪶 Summary")
        st.write(result["summary"].get("text", "No summary available."))
        
    # Sentiment
    if "sentiment" in result:
        sentiment = result["sentiment"]
        st.subheader("💭 Sentiment Analysis")
        st.write(f"**Sentiment:** {sentiment.get('sentiment', 'N/A')}")
        st.write(f"**Confidence:** {sentiment.get('confidence', 'N/A')}")
        st.write(f"**Emotions:** {', '.join(sentiment.get('emotions', [])) or 'N/A'}")
        st.write(f"**Rationale:** {sentiment.get('rationale', 'N/A')}")
    
    # Intent & Purpose
    if "intent_and_purpose" in result:
        intent = result["intent_and_purpose"]
        st.subheader("🎯 Intent & Purpose")
        st.write(f"**Intent:** {intent.get('intent', 'N/A')}")
        st.write(f"**Purpose:** {intent.get('purpose', 'N/A')}")
        st.write(f"**Rationale:** {intent.get('rationale', 'N/A')}")
    
    # Author Stance
    if "stance" in result:
        st.subheader("🧠 Author Stance")
        stance = result["stance"]
        st.write(stance if isinstance(stance, str) else json.dumps(stance, indent=2))

    # Knowledge Graph
    if "knowledge_graph" in result and result["knowledge_graph"]:
        st.subheader("🌟 Knowledge Graph")
        kg = result["knowledge_graph"]
        
        try:
            G = nx.DiGraph()
            
            for triple in kg.get("triples", []):
                subj, rel, obj = triple["subject"], triple["relation"], triple["object"]
                G.add_edge(subj, obj, label=rel)
            
            plt.figure(figsize=(10,6))
            pos = nx.spring_layout(G, k=0.6, iterations=60, seed=42)
            
            # Draw nodes and edges
            nx.draw_networkx_nodes(G, pos, node_size=1200, alpha=0.9, linewidths=1.3, edgecolors="black",node_color="lightblue", node_shape="o")
            nx.draw_networkx_edges(G, pos, width=1.2, edge_color="gray", arrows=True, arrowsize=13, connectionstyle="arc3,rad=0.1")
            nx.draw_networkx_labels(G, pos, font_size=11, font_color="black", font_weight="bold")
            
            # Add labels
            edge_labels = nx.get_edge_attributes(G, "label")
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="darkred", font_size=10, label_pos=0.5)
            
            plt.axis("off")
            plt.tight_layout()
            st.pyplot(plt)
        except Exception as e:
            st.warning(f"Could not render graph: {e}")
        
    # Chatbot
    st.header("💬 Chat with Knowledge Base")
    
    # Initialize memory and chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
 
    # Initialize display of chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
        
    user_query = st.text_input("Enter your question:")
    
    if st.button("Ask"):
        if user_query.strip():
            intent = classify_query_intent(user_query)
            response = None
            
            # Load vector store
            vector_store = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

            # Add contextual compression
            compressor = LLMChainExtractor.from_llm(llm)
            retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
            
            # Persistent conversational chain with memory (per session)
            if "chat_chain" not in st.session_state:
                st.session_state["chat_chain"] = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=st.session_state["chat_memory"],
                    return_source_documents=False
                )
               
            # Generate response         
            if intent == "knowledge":
                try:    
                    response = st.session_state["chat_chain"].invoke({"question": user_query})
                    response = response["answer"]
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            elif intent == "personal":
                # Retrieve personal facts only
                facts = recall_relevant_facts(user_query)
                context = "\n".join(facts)
                full_query = f"User personal context:\n{context}\n\nUser asked: {user_query}"
                response = llm.invoke(full_query).content.strip()
            else:
                facts = recall_relevant_facts(user_query)
                personal_context = "\n".join(facts)
                knowledge_context = retriever.invoke(user_query)
                combined_context = f"Personal context:\n{personal_context}\n\nKnowledge context:\n{knowledge_context}\n\nUser asked: {user_query}"
                response = llm.invoke(combined_context).content.strip()
            
            st.session_state["chat_history"].append({"role": "user", "content": user_query})
            st.session_state["chat_history"].append({"role": "assistant", "content": response})
            
    # Display chat history
    if st.session_state["chat_history"]:
        for msg in st.session_state["chat_history"]:
            role = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(f"**{role} {msg['role'].capitalize()}:** {msg['content']}")