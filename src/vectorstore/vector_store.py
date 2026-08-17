import chromadb
from chromadb.config import Settings
import tempfile
import os


VECTORSTORE_PATH = os.path.join(
    tempfile.gettempdir(),
    "vectorstore"
)


client = chromadb.PersistentClient(
    path=VECTORSTORE_PATH,
    settings=Settings(allow_reset=True)
)


collection = client.get_or_create_collection(
    name="documents"
)


def add_chunks_to_store(chunks, embeddings, filename):

    ids = [
        f"{filename}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": filename,
            "chunk_index": i
        }
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )


def get_document_count():
    return collection.count()


def get_sources():
    """
    Retourne les documents actuellement présents
    dans ChromaDB.
    """

    results = collection.get(
        include=["metadatas"]
    )

    sources = set()

    for metadata in results["metadatas"]:

        if metadata and "source" in metadata:
            sources.add(metadata["source"])

    return list(sources)


def delete_document(filename):
    """
    Supprime tous les chunks appartenant
    à un document.
    """

    collection.delete(
        where={
            "source": filename
        }
    )