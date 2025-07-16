import os
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from langgraph.graph import END, StateGraph


# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check and print status
print("API key loaded successfully!" if GOOGLE_API_KEY else "API key not found!")

# make sure to set the GOOGLE_API_KEY in your .env file
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.5,
    google_api_key=GOOGLE_API_KEY
)
# making sure llm is working add a print statement
print("LLM initialized successfully!")


@dataclass
class AgentState:
    query: Optional[str] = None
    retrieved_docs: Optional[List[Document]] = None
    answer: Optional[str] = None


# raissing an error if pdf is not found
if not os.path.exists("data"):
    raise FileNotFoundError("The 'data' directory does not exist. Please create it and add your PDF files.")

# Setup embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load all PDFs from folder
pdf_dir = "data"
all_pdf = []

for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_dir, filename))
        all_pdf.extend(loader.load())

print("All PDFs loaded and parsed:", len(all_pdf), "documents.")

# Now split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(all_pdf)
print("Chunks created:", len(chunks))

persist_directory = r"C:\Users\Tanmmay R Joseph\OneDrive\Desktop\RagAgent"
collection_name = "rag_agent_test"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=collection_name,
    persist_directory=persist_directory
)

print("✅ Vector DB created and persisted.")

# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


# defining input and updating the query
def input_node(s: AgentState):
    """ This function prompts the user for a question and updates the state with the query."""
    s.query = input("What is your question? ")
    return s


def retriever_node(s: AgentState):
    """
    Retrieves relevant documents based on the user's query using Chroma vector store.
    Embeds the query and performs vector similarity search to populate AgentState.retrieved_docs.
    """

    if not s.query:
        raise ValueError("❌ Query is not set. Please provide a valid query.")

    # Step 1: Embed the query using Google Generative AI Embeddings
    query_embedding = embeddings.embed_query(s.query)

    # Step 2: Load existing Chroma vector store
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    # Step 3: Perform similarity search
    try:
        results = vector_store.similarity_search_by_vector(query_embedding, k=5)
    except Exception as e:
        raise RuntimeError(f"⚠️ Error during similarity search: {e}")

    # Step 4: Handle empty results gracefully
    if not results:
        print("⚠️ No relevant documents found for this query.")
        s.retrieved_docs = [Document(page_content="No relevant content found.", metadata={"source": "None"})]
    else:
        s.retrieved_docs = results
        print(f"✅ Retrieved {len(s.retrieved_docs)} relevant documents.")

    return s


def synthesis_node(s: AgentState):
    """Uses Gemini to generate an answer from retrieved documents."""

    if not s.retrieved_docs or not s.query:
        raise ValueError("❌ Missing retrieved documents or query.")

    # Step 1: Combine retrieved chunks into a single context
    context = "\n\n".join([doc.page_content for doc in s.retrieved_docs])

    # Step 2: Format the prompt
    prompt = f"""You are a helpful assistant.
Answer the question using ONLY the following context:

{context}

Question: {s.query}
Answer:"""

    # Step 3: Call Gemini using .invoke()
    s.answer = llm.invoke(prompt)

    print("✅ Answer generated.")
    return s


def output_node(s: AgentState):
    """Outputs the final answer to the user."""
    if not s.answer:
        raise ValueError("No answer generated. Please check the previous steps.")

    print(f"\n💡 Final Answer:\n{s.answer}")
    return s  # ✅ Must return updated state, not END


# Create the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("input", input_node)
graph.add_node("retriever", retriever_node)
graph.add_node("synthesis", synthesis_node)
graph.add_node("output", output_node)

# Define transitions
graph.add_edge("input", "retriever")
graph.add_edge("retriever", "synthesis")
graph.add_edge("synthesis", "output")
graph.add_edge("output", END)

# Set entry and exit points
graph.set_entry_point("input")
graph.set_finish_point("output")

# Compile the app
app = graph.compile()

initial_state = {
    "query": "are women safe in public transport like taxi ?"
}

# Run the app
final_state = app.invoke(initial_state)

# Print answer and source snippets
print("\n✅ Final Answer:", final_state["answer"])

print("\n📚 Retrieved Docs Preview:")
for i, doc in enumerate(final_state["retrieved_docs"]):
    print(f"Doc {i + 1} — Source: {doc.metadata.get('source', 'N/A')}")
    print(doc.page_content[:200], "...\n")
