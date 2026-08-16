from src.vectorstore.vector_store import collection
from src.embeddings.embedding_service import create_embeddings


def search_similar_chunks(question, k=3):
    """
    Recherche les chunks les plus similaires à une question.
    """

    # Créer l'embedding de la question
    question_embedding = create_embeddings([question])

    # Recherche dans ChromaDB
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=k
    )

    return results


