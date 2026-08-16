import streamlit as st

from src.loaders.document_loader import extract_text
from src.chunking.text_splitter import split_text_into_chunks
from src.embeddings.embedding_service import create_embeddings
from src.vectorstore.vector_store import add_chunks_to_store , get_document_count
from src.retrieval.retriever import search_similar_chunks
from src.llm.groq_client import generate_answer





st.set_page_config(
    page_title="DocuChat AI",
    page_icon="📚",
    layout="wide"
)


st.title("📚 DocuChat AI")

st.write(
    "Téléversez un document PDF, TXT ou DOCX."
)


uploaded_file = st.file_uploader(
    "Choisissez un fichier",
    type=["pdf", "txt", "docx"]
)


if uploaded_file is not None:

    st.success(
        f"📄 Fichier chargé : {uploaded_file.name}"
    )

    try:

        # =====================================
        # PHASE 1 : EXTRACTION
        # =====================================

        content = extract_text(uploaded_file)

        if not content:
            st.warning(
                "Aucun texte n'a pu être extrait."
            )

        else:

            st.subheader("📖 Texte extrait")

            st.text_area(
                "Contenu du document",
                content,
                height=300,
                key="extracted_text"
            )


            # =====================================
            # PHASE 2 : CHUNKING
            # =====================================

            chunks = split_text_into_chunks(
                content,
                chunk_size=500,
                chunk_overlap=100
            )

            st.subheader("✂️ Chunks")

            st.write(
                f"Nombre de chunks créés : **{len(chunks)}**"
            )


            for i, chunk in enumerate(chunks):

                with st.expander(
                    f"Chunk {i + 1} — {len(chunk)} caractères"
                ):

                    st.write(chunk)


            # =====================================
            # PHASE 3 : EMBEDDINGS
            # =====================================

            st.subheader("🧠 Embeddings")

            embeddings = create_embeddings(chunks)

            st.write(
                f"Nombre de vecteurs : **{len(embeddings)}**"
            )

            st.write(
                f"Dimension d'un vecteur : "
                f"**{embeddings.shape[1]}**"
            )


            # Afficher le premier embedding
            st.write("Premier embedding :")

            st.write(
                embeddings[0]
            )
            
            # =====================================
            # PHASE 4 : VECTOR STORE - CHROMADB
            # =====================================

            st.subheader("🗄️ ChromaDB")

            add_chunks_to_store(
                chunks,
                embeddings,
                uploaded_file.name
            )

            st.success(
                "✅ Chunks et embeddings ajoutés à ChromaDB."
            )

            st.write(
                f"Nombre de documents stockés dans ChromaDB : "
                f"**{get_document_count()}**"
            )
            # =====================================
            # PHASE 5 : RETRIEVAL
            # =====================================

            st.subheader("🔎 Recherche sémantique")

            question = st.text_input(
                "Posez une question sur votre document :"
            )

            if question:

                results = search_similar_chunks(
                    question,
                    k=5
                )

                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]

                # =====================================
                # PHASE 6 : GÉNÉRATION LLM
                # =====================================

                st.subheader("🤖 Réponse de l'IA")

                answer = generate_answer(question, documents)

                st.write(answer)

                 # =====================================
                # PHASE 7 : CITATION DES SOURCES
                # =====================================

                st.write("### 📚 Sources utilisées")

                for i in range(len(documents)):
                    with st.expander(f"Source {i+1} — {metadatas[i]['source']} (score : {distances[i]:.4f})"):
                        st.write(documents[i])
    except Exception as e:

        st.error(
            f"Erreur : {e}"
        )

        st.exception(e)

