import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./vectorstore",
    settings=Settings(allow_reset=True)
)

collection = client.get_or_create_collection(name="documents")


def add_chunks_to_store(chunks, embeddings, filename):
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    documents = chunks
    metadatas = [
        {"source": filename, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

def get_document_count():
    return collection.count()