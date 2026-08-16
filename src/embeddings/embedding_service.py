from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


model = SentenceTransformer(MODEL_NAME)


def create_embeddings(chunks):
    """
    Transforme les chunks en vecteurs numériques.

    Args:
        chunks: Liste de textes.

    Returns:
        Liste des embeddings.
    """

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    return embeddings