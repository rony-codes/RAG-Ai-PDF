from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore



embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)



vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model,
)



llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)



user_query = input("Ask Something: ")


search_results = vector_db.similarity_search(
    query=user_query,
    k=4
)


context = "\n\n---\n\n".join(
    [
        f"""
Page Content:
{result.page_content}

Page Number:
{result.metadata.get("page_label", "Unknown")}

File Location:
{result.metadata.get("source", "Unknown")}
"""
        for result in search_results
    ]
)




SYSTEM_PROMPT = f"""
You are a helpful AI assistant answering questions about a PDF document.

You must answer the user's question using ONLY the retrieved context below.

Rules:

1. Do not use outside knowledge.

2. If the answer cannot be found in the provided context, say:
   "I could not find this information in the PDF."

3. Mention the PDF page number where the information was found.

4. Keep the answer clear and concise.

5. Do not invent information.

Retrieved Context:

{context}
"""


# ---------------------------------------
# 8. SEND QUESTION TO LOCAL OLLAMA MODEL
# ---------------------------------------

response = llm.invoke(
    [
        ("system", SYSTEM_PROMPT),
        ("human", user_query),
    ]
)


# ---------------------------------------
# 9. PRINT ANSWER
# ---------------------------------------

print(f"\n🤖: {response.content}")