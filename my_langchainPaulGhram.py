

import re
import os, requests, shutil
from pathlib import Path
from bs4 import BeautifulSoup
from unstructured.partition.html import partition_html
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------
os.environ["USER_AGENT"] = "ZakRAGBot/1.0 (+mailto:you@example.com)"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
BASE_URL = "https://www.paulgraham.com/"
INDEX_URL = BASE_URL + "articles.html"

DB_DIR = Path("local_db")
EXPORT_DIR = Path("exported_texts")
EXPORT_DIR.mkdir(exist_ok=True)

REBUILD = False  # re-fetch and re-embed everything


# ------------------------------------------------------------
# 2. HELPERS
# ------------------------------------------------------------
def safe_filename(title: str) -> str:
    """Convert any string into a safe Windows-compatible filename."""
    name = re.sub(r'[<>:"/\\|?*]', "_", title)
    name = re.sub(r"\s+", "_", name.strip())
    if len(name) > 120:
        name = name[:120]
    return name


def get_all_article_links(index_url: str):
    """Fetch all essay URLs from the Paul Graham index page."""
    print(f"Scraping index page: {index_url}")
    headers = {"User-Agent": os.environ["USER_AGENT"]}
    html = requests.get(index_url, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")

    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Only keep essay pages
        if href.endswith(".html") and not href.startswith("articles"):
            urls.append(BASE_URL + href)

    urls = sorted(set(urls))
    print(f"✅ Found {len(urls)} articles.")
    return urls


def partition_web_page(url: str):
    """Extract HTML using Unstructured to get high-quality parsing."""
    print(f"Partitioning web page: {url}")
    headers = {"User-Agent": os.environ["USER_AGENT"]}
    html_text = requests.get(url, headers=headers).text

    elements = partition_html(text=html_text)
    print(f"Extracted {len(elements)} elements")

    soup = BeautifulSoup(html_text, "html.parser")
    title = soup.title.string.strip() if soup.title else "Untitled"
    author = "Paul Graham"
    date_meta = soup.find("meta", {"name": "date"})
    date = date_meta["content"] if date_meta else "Unknown"

    docs = []
    for el in elements:
        text = str(el.text).strip()
        if len(text) > 30:
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "category": el.category or "Unknown",
                        "title": title,
                        "author": author,
                        "date": date,
                        "source": url,
                    },
                )
            )

    print(f"{len(docs)} usable text sections extracted ({title})")

    # Export parsed text (for inspection)
    safe_title = safe_filename(title)
    export_path = EXPORT_DIR / f"{safe_title}_parsed.txt"
    with open(export_path, "w", encoding="utf-8") as f:
        for i, d in enumerate(docs, 1):
            meta = d.metadata
            f.write(f"\n--- Section {i} ---\n")
            f.write(f"Title: {meta.get('title')}\n")
            f.write(f"Author: {meta.get('author')}\n")
            f.write(f"Date: {meta.get('date')}\n")
            f.write(f"Source: {meta.get('source')}\n\n")
            f.write(d.page_content.strip())
            f.write("\n---------------------------\n")

    return docs


# ------------------------------------------------------------
# 3. DATA INGESTION + CHUNKING
# ------------------------------------------------------------
if not DB_DIR.exists() or REBUILD:
    if REBUILD and DB_DIR.exists():
        print("🧹 Removing old vectorstore...")
        shutil.rmtree(DB_DIR)

    print("📡 Fetching and parsing all Paul Graham essays...")
    URLS = get_all_article_links(INDEX_URL)
    all_docs = []

    for url in URLS:
        try:
            all_docs.extend(partition_web_page(url))
        except Exception as e:
            print(f"❌ Error parsing {url}: {e}")

    print(f"📚 Total documents extracted: {len(all_docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_docs = splitter.split_documents(all_docs)
    print(f"🧩 Total chunks prepared: {len(chunked_docs)}")

    print("🧠 Building vectorstore from scratch...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(chunked_docs, embedding=embeddings, persist_directory=str(DB_DIR))

    print(f"✅ Vectorstore built and persisted with {len(chunked_docs)} chunks.")

else:
    print("📦 Loading existing vectorstore...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)


# ------------------------------------------------------------
# 4. RETRIEVAL + ANSWERING
# ------------------------------------------------------------
llm = OllamaLLM(model=LLM_MODEL)

system_prompt = """You are an expert summarizer of Paul Graham’s essays.
Answer factually and concisely, preserving the author’s reasoning and examples.
Use direct quotes sparingly, attribute them, and end every answer with:
(Source: <title> — <URL>).
Never invent links or content not present in the context."""

def ask(question: str):
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 7, "score_threshold": 0.1},
    )

    retrieved_docs = retriever.invoke(question)
    if not retrieved_docs:
        print("⚠️ No docs passed threshold; falling back to similarity search.")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(question)

    print("\n🔎 Retrieved essays:")
    for i, d in enumerate(retrieved_docs[:5], 1):
        print(f"  {i}. {d.metadata.get('title')}")

    context = "\n".join(
        f"[From: {d.metadata.get('title')} — {d.metadata.get('source')}]\n{d.page_content}"
        for d in retrieved_docs
    )

    prompt = f"""{system_prompt}

Question:
{question}

Context:
{context}
"""
    response = llm.invoke(prompt)
    print("\n💬", response)


# ------------------------------------------------------------
# 5. CHAT LOOP
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n🤖 RAG Chatbot Ready!")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        ask(q)
