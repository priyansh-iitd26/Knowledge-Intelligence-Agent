import os, warnings, re, json
warnings.filterwarnings("ignore")
from typing import List, Dict, Any, TypedDict
from langchain_community.document_loaders import WebBaseLoader, PDFPlumberLoader, YoutubeLoader, Docx2txtLoader, UnstructuredPowerPointLoader, TextLoader, CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import networkx as nx
from uuid import uuid4
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from langgraph.checkpoint.memory import MemorySaver
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import streamlit as st

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
if not OPENAI_API_KEY:
    st.error("Please use your OpenAI API key to execute the workflow.")
    st.stop()
if not NEO4J_URI:
    st.error("Please use your Neo4j URI to execute the workflow.")
    st.stop()
if not NEO4J_USER:
    st.error("Please use your Neo4j user to execute the workflow.")
    st.stop()
if not NEO4J_PASSWORD:
    st.error("Please use your Neo4j password to execute the workflow.")
    st.stop()

# Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Initialize LangChain objects
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, api_key=OPENAI_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", "?", "!", " ", ""]
)
persist_dir = "./"

class MultiContentKnowledgeState(TypedDict):
    url: str
    file_path: str
    source_type: str
    content_id: str
    raw_text: str
    cleaned_text: str
    embeddings_indexed: bool
    summary: Dict[str, Any]
    sentiment: Dict[str, Any]
    intent_and_purpose: Dict[str, Any]
    stance: Dict[str, Any]
    knowledge_graph: Dict[str, Any]
    final_report: Dict[str, Any]
    
class Sentiment(BaseModel):
    sentiment: str = Field(..., description="Overall sentiment: positive, neutral, or negative.")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1.")
    emotions: List[str] = Field(default_factory=list, description="Detected emotions.")
    rationale: str = Field(default="", description="Brief reasoning behind the sentiment classification.")
    
sentiment_llm = llm.with_structured_output(Sentiment)

class AuthorStance(BaseModel):
    stance: str = Field(..., description="Main stance of the author: supportive, oppositional, neutral, mixed, skeptical, advocacy, exploratory.")
    
stance_llm = llm.with_structured_output(AuthorStance)

class IntentAndPurposeResult(BaseModel):
    intent: str = Field(..., description="Overall intent: positive, neutral, or negative.")
    purpose: str = Field(..., description="Overall purpose: objective, subjective, or unknown.")
    rationale: str = Field(..., description="Brief reasoning behind the intent classification.")
    
intent_llm = llm.with_structured_output(IntentAndPurposeResult)

class KnowledgeTriple(BaseModel):
    subject: str = Field(..., description="The main entity or concept (e.g., 'Tesla')")
    relation: str = Field(..., description="The relationship type (e.g., 'develops')")
    object: str = Field(..., description="The secondary entity or concept (e.g., 'electric vehicles')")
    
class KnowledgeTriples(BaseModel):
    triples: List[KnowledgeTriple]

triplet_llm = llm.with_structured_output(KnowledgeTriples)

def to_serializable(obj):
    """Convert objects to JSON-serializable"""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    elif hasattr(obj, "dict"):
        return to_serializable(obj.dict())
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    else:
        return str(obj)
    
def download_temp(url: str, suffix: str) -> str:
    """Download a remote file to a temporary path for processing."""
    import requests, tempfile
    response = requests.get(url)
    response.raise_for_status()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(response.content)
    tmp_file.close()
    return tmp_file.name

def extract_content(state: MultiContentKnowledgeState) -> dict:
    url = state.get("url")
    file_path = state.get("file_path")
    source_type = state.get("source_type", "web")

    if not url and not file_path:
        return {}

    # Detect type
    if url:
        lower_url = url.lower()
        if "youtube.com" in lower_url or "youtu.be" in lower_url:
            source_type = "youtube"
        elif lower_url.endswith(".pdf"):
            source_type = "pdf"
        elif lower_url.endswith(".docx"):
            source_type = "docx"
        elif lower_url.endswith(".pptx"):
            source_type = "pptx"
        elif lower_url.endswith(".csv"):
            source_type = "csv"
        elif lower_url.endswith(".json"):
            source_type = "json"
        elif lower_url.endswith(".txt"):
            source_type = "txt"
        elif lower_url.endswith((".html", ".htm")):
            source_type = "html"
        elif "arxiv.org" in lower_url:
            source_type = "arxiv"
        elif "notion.so" in lower_url:
            source_type = "notion"
        else:
            source_type = "web"
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
        source_type = ext_map.get(ext, "web")

    # Loaders
    if source_type == "youtube":
        video_id_match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
        if not video_id_match:
            raise ValueError(f"Invalid YouTube URL: {url}")
        loader = YoutubeLoader(video_id_match.group(1))
    elif source_type in ["web", "html"]:
        loader = WebBaseLoader(url)

    elif source_type == "pdf":
        if url:
            tmp_path = download_temp(url, ".pdf")
            loader = PDFPlumberLoader(tmp_path)
        else:
            loader = PDFPlumberLoader(file_path)

    elif source_type == "docx":
        if url:
            tmp_path = download_temp(url, ".docx")
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = Docx2txtLoader(file_path)

    elif source_type == "pptx":
        if url:
            tmp_path = download_temp(url, ".pptx")
            loader = UnstructuredPowerPointLoader(tmp_path)
        else:
            loader = UnstructuredPowerPointLoader(file_path)

    elif source_type == "csv":
        if url:
            tmp_path = download_temp(url, ".csv")
            loader = CSVLoader(tmp_path)
        else:
            loader = CSVLoader(file_path)

    elif source_type == "json":
        if url:
            tmp_path = download_temp(url, ".json")
            loader = JSONLoader(tmp_path, jq_schema=".", text_content=False)
        else:
            loader = JSONLoader(file_path, jq_schema=".", text_content=False)

    elif source_type == "txt":
        if url:
            tmp_path = download_temp(url, ".txt")
            loader = TextLoader(tmp_path)
        else:
            loader = TextLoader(file_path)

    else:
        raise ValueError(f"Unsupported or unknown source type: {source_type}")

    content_id = f"ct_{uuid4().hex}"
    documents = loader.load()

    return {
        "url": url,
        "file_path": file_path,
        "source_type": source_type,
        "content_id": content_id,
        "raw_text": " ".join(doc.page_content for doc in documents),
        "embeddings_indexed": False
    }

def clean_and_preprocess_text(state: MultiContentKnowledgeState) -> dict:
    cleaned_text = re.sub(r"\n{2,}", "\n\n", state["raw_text"])  
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text).strip()
    return {"cleaned_text": cleaned_text}
            
def split_text_and_index(state: MultiContentKnowledgeState) -> dict:
    cleaned_text = state.get("cleaned_text", "")
    content_id = state.get("content_id", "")    
    
    # Split text into chunks and persist externally
    chunks = [chunk for chunk in text_splitter.split_text(cleaned_text) if chunk.strip()]
    if not chunks:
        raise ValueError("No valid text chunks found for embeddings.")
    
    # Create documents
    chunks = [c for c in text_splitter.split_text(cleaned_text) if c.strip()]
    if not chunks:
        return {"content_id": content_id, "chunk_count": 0, "embeddings_indexed": False}

    docs = [Document(page_content=chunk, metadata={"content_id": content_id}) for chunk in chunks]
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    embed_batch_size = 32  
    all_embeddings = []
    all_metadatas = []

    for i in range(0, len(texts), embed_batch_size):
        batch_texts = texts[i : i + embed_batch_size]
        batch_metas = metadatas[i : i + embed_batch_size]

        try:
            batch_embeddings = embeddings.embed_documents(batch_texts)
        except Exception as e:
            print(f"Embedding batch {i//embed_batch_size} failed: {e}")
            continue

        all_embeddings.extend(batch_embeddings)
        all_metadatas.extend(batch_metas)

    if not all_embeddings:
        return {"content_id": content_id, "chunk_count": len(chunks), "embeddings_indexed": False}

    text_embeddings = list(zip(texts[:len(all_embeddings)], all_embeddings))
    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embeddings, 
        embedding=embeddings,
        metadatas=all_metadatas
    )

    # Persist local index
    vector_store.save_local(persist_dir)

    return {
        "content_id": content_id,
        "chunk_count": len(chunks),
        "embeddings_indexed": True
    }
    
def generate_summary(state: MultiContentKnowledgeState) -> dict:
    content_id = state.get("content_id", "")
    vector_store = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    # Add contextual compression
    compressor = LLMChainExtractor.from_llm(llm)
    retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    
    # Retrieve top, relevant, and compressed chunks
    retrieval_query = "Fetch the most critical information across each chunk to be used for summarization."
    
    documents = retriever.invoke(retrieval_query)
    
    if not documents:
        return {"summary": {"text": "No relevant content found for summarization.", "sources": []}}
    
    map_prompt = PromptTemplate(template="""
        Given the following extracted parts of a long document: {text}
        Write a clear and concise summary of an individual part of the document.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    """, input_variables=["text"])
    combine_prompt = PromptTemplate(template="""
        Given summaries of all chunks and their credible sources of a long document: {text}
        Return a clear and concise summary of the document based on the individual summaries.
    """, input_variables=["text"])
    
    # Initialize hierarchical summarization (map-reduce)
    summarize_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce", # map -> summarize each chunk, reduce -> combine individual summaries
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        return_intermediate_steps=True
    )
    # Execute summarization
    result = summarize_chain.invoke({"input_documents": documents})

    # Extract main summary and individual summaries
    final_summary = result["output_text"]
    partial_summaries = result.get("intermediate_steps", [])
    
    # Maintain source provenance
    source_ids = [doc.metadata.get("source", doc.metadata.get("content_id", "unknown")) for doc in documents]
    
    return {
        "content_id": content_id,
        "summary": {
            "text": final_summary,
            "partial_summaries": partial_summaries,
            "sources": source_ids
        }
    }
        
def perform_sentiment_analysis(state: MultiContentKnowledgeState) -> dict:
    """
        Quickly analyze overall sentiment and emotional tone of content.
        Applies to all content types (YouTube, articles, PDFs, papers, etc.)
    """
    summary_data = state.get("summary", {})
    summary_text = (
        summary_data.get("text") if isinstance(summary_data, dict) else str(summary_data)
    )

    if not summary_text or not summary_text.strip():
        return {"sentiment": {"sentiment": "neutral", "confidence": 0.5, "emotions": [], "rationale": "Empty summary."}}

    prompt = f"""
        Determine the overall sentiment and emotional tone of this text.
        Respond in JSON with:
        {{
        "sentiment": "positive" | "neutral" | "negative",
        "confidence": 0-1,
        "emotions": ["joy", "anger", "fear", "trust", ...],
        "rationale": "brief explanation"
        }}
        Text:
        {summary_text}
    """
    
    try:
        response = sentiment_llm.invoke(prompt)
        sentiment = response.dict()
    except Exception:
        sentiment = {"sentiment": "neutral", "confidence": 0.5, "emotions": [], "rationale": "Fallback neutral sentiment."}
    
    return {
        "sentiment": sentiment
    }

def extract_intent_and_purpose(state: MultiContentKnowledgeState) -> dict:
    summary_data = state.get("summary", {})
    summary_text = (
        summary_data.get("text") if isinstance(summary_data, dict) else str(summary_data)
    )

    if not summary_text or not summary_text.strip():
        return {"intent_and_purpose": {"intent": "unknown", "purpose": "unknown", "rationale": "No text found."}}

    prompt = f"""
        Determine the main intent and purpose of this text.
        Respond in JSON with:
        {{
            "intent": "positive" | "neutral" | "negative",
            "purpose": "objective" | "subjective" | "unknown",
            "rationale": "brief explanation"
        }}
        Text:
        {summary_text}
    """

    try:
        response = intent_llm.invoke(prompt)
        intent_and_purpose = response.dict()
    except Exception:
        intent_and_purpose = {"intent": "neutral", "purpose": "unknown", "rationale": "Fallback unknown intent."}

    return {"intent_and_purpose": intent_and_purpose}
    
def analyze_stance(state: MultiContentKnowledgeState) -> dict:
    summary_data = state.get("summary", {})
    summary_text = (
        summary_data.get("text") if isinstance(summary_data, dict) else str(summary_data)
    )
    if not summary_text or not summary_text.strip():
        return {"stance": "unknown", "confidence": 0.0, "reason": "Empty summary."}
    
    prompt = f"""
        Determine main stance of the author of this text.
        Respond in JSON with:
        {{
            "stance": "supportive" | "oppositional" | "neutral" | "mixed" | "skeptical" | "advocacy" | "exploratory"
        }}
        \nText: {summary_text}
    """
    
    try:
        response = stance_llm.invoke(prompt)
        stance = response.dict()
    except Exception:
        stance = {"stance": "neutral"}
    
    return {
        "stance": stance["stance"]
    }
    
def extract_citations(state: MultiContentKnowledgeState) -> dict:
    vector_store = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 25})
    query = "Find text segments that reference other studies, authors, or papers."
    docs = retriever.invoke(query)
    
    text_segments = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
        Extract all citations or references from the following text segments.
        Each citation should have: authors, year, title (if any), type (journal, book, arXiv, report, etc.), and URL/DOI if mentioned.
        Respond in JSON list.
        Segments:
        {text_segments}
    """
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        citations = json.loads(re.search(r"\[.*]", content, re.S).group(0))
    except Exception:
        citations = []
    
    return {
        "citations": citations
    }

def create_knowledge_graph(state: MultiContentKnowledgeState) -> dict:
    """
        Generates a knowledge graph from the summary text using structured LLM outputs.
        Extracts (subject, relation, object) triplets and persists the graph in Neo4j.
    """
    summary_data = state.get("summary", {})
    summary_text = summary_data.get("text") if isinstance(summary_data, dict) else summary_data

    if not summary_text or not summary_text.strip():
        return {"knowledge_graph": None}

    prompt = f"""
        Extract factual (subject, relation, object) triples from the following summary text.
        Ensure each triple expresses a meaningful semantic relationship between entities.
        Text:
        {summary_text}
    """

    try:
        triples = triplet_llm.invoke(prompt)
        triples_output = triplet_llm.invoke(prompt)

        # Handle multiple return formats gracefully
        if isinstance(triples_output, KnowledgeTriples):
            triples = [t.dict() for t in triples_output.triples]

        elif isinstance(triples_output, tuple):
            # Sometimes returns (KnowledgeTriples, other_info)
            triples_model = triples_output[0]
            if hasattr(triples_model, "triples"):
                triples = [t.dict() for t in triples_model.triples]
            elif isinstance(triples_model, list):
                triples = [t.dict() if hasattr(t, "dict") else t for t in triples_model]
            else:
                triples = []

        elif isinstance(triples_output, list):
            triples = [t.dict() if hasattr(t, "dict") else t for t in triples_output]

        elif isinstance(triples_output, dict) and "triples" in triples_output:
            triples = triples_output["triples"]

        else:
            triples = []
    except Exception as e:
        print(f"LLM structured output failed: {e}")
        triples = []

    # Build in-memory graph using NetworkX
    G = nx.DiGraph()
    
    for triple in triples:
        subj, rel, obj = triple["subject"], triple["relation"], triple["object"]
        G.add_node(subj, type="entity")
        G.add_node(obj, type="entity")
        G.add_edge(subj, obj, relation=rel)

    # Persist graph to Neo4j
    with driver.session() as session:
        for triple in triples:
            subj, rel, obj = triple["subject"], triple["relation"], triple["object"]
            query = """
            MERGE (s:Entity {name: $subj})
            MERGE (o:Entity {name: $obj})
            MERGE (s)-[r:RELATION {type: $rel}]->(o)
            RETURN s, r, o
            """
            try:
                session.run(query, subj=subj, obj=obj, rel=rel)
            except Exception as e:
                print(f"Neo4j persistence failed for triple {triple}: {e}")

    # Convert to serializable JSON output
    edges = [{"source": u, "target": v, "relation": d["relation"]} for u, v, d in G.edges(data=True)]

    return {
        "knowledge_graph": {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "triples": triples,
            "edges": edges
        }
    }
    
def route_for_citations(state: MultiContentKnowledgeState) -> str:
    """ Route only research-type content to citation extraction. """
    source_type = state.get("source_type", "")
    
    if source_type in ["arxiv", "paper", "pdf"]:
        return "extract_citations"
    else:
        return "create_knowledge_graph"

def finalize_outputs(state: MultiContentKnowledgeState) -> dict:
    """ Aggregate all results (summary, sentiment, stance, intent, citations, KG) into a structured final output. """
    summary = state.get("summary", {}).get("text", "No summary available.")
    sentiment = state.get("sentiment", {})
    intent = state.get("intent_and_purpose", {})
    stance = state.get("stance", "N/A")
    
    final_report = f"""
        Multi-Source Knowledge Intelligence Agent Report

        Summary:
        
        {summary}

        Sentiment Analysis:
        
        Sentiment: {sentiment.get("sentiment", "N/A")}  
        Confidence: {sentiment.get("confidence", "N/A")}  
        Emotions: {", ".join(sentiment.get("emotions", [])) if sentiment.get("emotions") else "N/A"}  
        Rationale: {sentiment.get("rationale", "N/A")}

        Intent & Purpose:
        
        Intent: {intent.get("intent", "N/A")}  
        Purpose: {intent.get("purpose", "N/A")}  
        Rationale: {intent.get("rationale", "N/A")}
        
        Knowledge Graph:
        
        Node Count: {state.get("knowledge_graph", {}).get("node_count", "N/A")}  
        Edge Count: {state.get("knowledge_graph", {}).get("edge_count", "N/A")}

        Stance:
        {stance}
    """

    return {"final_report": final_report.strip()}
    
# Create a StateGraph object
graph = StateGraph(MultiContentKnowledgeState)

# Core pipeline
graph.add_node("extract_content", extract_content)
graph.add_node("clean_and_preprocess_text", clean_and_preprocess_text)
graph.add_node("split_text_and_index", split_text_and_index)
graph.add_node("generate_summary", generate_summary)

# Analysis modules
graph.add_node("analyze_sentiment", perform_sentiment_analysis)
graph.add_node("extract_intent_and_purpose", extract_intent_and_purpose)
graph.add_node("analyze_stance", analyze_stance)

# Conditional and parallel branches
graph.add_node("extract_citations", extract_citations)
graph.add_node("create_knowledge_graph", create_knowledge_graph)

graph.add_node("merge_for_final", lambda state: state)

# Final aggregator
graph.add_node("finalize_outputs", finalize_outputs)

# Edges
graph.add_edge(START, "extract_content")
graph.add_edge("extract_content", "clean_and_preprocess_text")
graph.add_edge("clean_and_preprocess_text", "split_text_and_index")
graph.add_edge("split_text_and_index", "generate_summary")

# Parallel branches from summary
for node in ["analyze_sentiment", "extract_intent_and_purpose", "analyze_stance"]:
    graph.add_edge("generate_summary", node)
    
# Conditional branching for citations or direct knowledge graph
graph.add_conditional_edges(
    "generate_summary",
    route_for_citations,
    {
        "extract_citations": "extract_citations",
        "create_knowledge_graph": "create_knowledge_graph"
    }
)

graph.add_edge("extract_citations", "create_knowledge_graph")

for node in ["analyze_sentiment", "extract_intent_and_purpose", "analyze_stance", "create_knowledge_graph"]:
    graph.add_edge(node, "merge_for_final")
    
graph.add_edge("merge_for_final", "finalize_outputs")
graph.add_edge("finalize_outputs", END)

memory = MemorySaver()

# Compile the graph
workflow = graph.compile(checkpointer=memory)

# # Run the workflow
# final_result = workflow.invoke({
#     "url": "https://github.com/resources/articles/devops/ci-cd",
#     "source_type": "html"
# }, config={"configurable": {"thread_id": "knowledge_intelligence_workflow_id"}})