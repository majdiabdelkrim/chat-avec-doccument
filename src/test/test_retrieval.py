from src.retrieval.retriever import search_similar_chunks


question = "Qu'est-ce qu'un RAG ?"

results = search_similar_chunks(question, k=3)

print(results)