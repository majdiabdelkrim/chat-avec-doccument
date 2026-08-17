from src.vectorstore.vector_store import collection


def search_similar_chunks(
    question,
    k=5,
    source=None
):
    """
    TEST DE CLASSEMENT

    Récupère les chunks présents dans ChromaDB
    et leur attribue un score artificiel.

    Exemple :
        10 -> premier
         9 -> deuxième
         8 -> troisième
         7 -> quatrième
    """

    # =====================================
    # RÉCUPÉRER TOUS LES CHUNKS
    # =====================================

    results = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = results["documents"]
    metadatas = results["metadatas"]

    # =====================================
    # FILTRER PAR DOCUMENT SI NÉCESSAIRE
    # =====================================

    if source is not None:

        filtered_documents = []
        filtered_metadatas = []

        for i in range(len(documents)):

            if metadatas[i]["source"] == source:

                filtered_documents.append(
                    documents[i]
                )

                filtered_metadatas.append(
                    metadatas[i]
                )

        documents = filtered_documents
        metadatas = filtered_metadatas

    # =====================================
    # CRÉER UN SCORE DE TEST
    # =====================================

    results_for_test = []

    score = len(documents)

    for i in range(len(documents)):

        results_for_test.append(
            {
                "document": documents[i],
                "metadata": metadatas[i],
                "score": score
            }
        )

        score -= 1

    # =====================================
    # TRI DU PLUS GRAND SCORE
    # =====================================

    results_for_test.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # =====================================
    # LIMITER À K RÉSULTATS
    # =====================================

    results_for_test = results_for_test[:k]

    # =====================================
    # AFFICHAGE TEST
    # =====================================

    print("\n===== CLASSEMENT TEST =====")

    for i, result in enumerate(results_for_test):

        print(
            f"{i + 1}. "
            f"Score : {result['score']} | "
            f"Source : {result['metadata']['source']}"
        )

    # =====================================
    # FORMAT ATTENDU PAR MAIN.PY
    # =====================================

    return {
        "documents": [
            [result["document"] for result in results_for_test]
        ],
        "metadatas": [
            [result["metadata"] for result in results_for_test]
        ],
        "distances": [
            [result["score"] for result in results_for_test]
        ]
    }