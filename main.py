import streamlit as st

from src.loaders.document_loader import extract_text
from src.chunking.text_splitter import split_text_into_chunks
from src.embeddings.embedding_service import create_embeddings
from src.vectorstore.vector_store import (
    add_chunks_to_store,
    get_document_count
)
from src.retrieval.retriever import search_similar_chunks
from src.llm.groq_client import generate_answer


# =====================================
# CONFIGURATION STREAMLIT
# =====================================

st.set_page_config(
    page_title="DocuChat AI",
    page_icon="📚",
    layout="wide"
)


# =====================================
# SESSION STATE
# =====================================

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================
# TITRE
# =====================================

st.title("📚 DocuChat AI")

st.write(
    "Téléversez un ou plusieurs documents (PDF, TXT, DOCX) "
    "et posez vos questions."
)


# =====================================
# UPLOAD DOCUMENTS
# =====================================

uploaded_files = st.file_uploader(
    "Choisissez un ou plusieurs fichiers",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)


# =====================================
# TRAITEMENT DES NOUVEAUX DOCUMENTS
# =====================================

if uploaded_files:

    for uploaded_file in uploaded_files:

        # Ne traiter que les fichiers pas encore indexés
        if uploaded_file.name not in st.session_state.processed_files:

            with st.status(
                f"Traitement de {uploaded_file.name}...",
                expanded=True
            ) as status:

                try:

                    # =====================================
                    # PHASE 1 : EXTRACTION
                    # =====================================

                    content = extract_text(uploaded_file)

                    if not content:
                        st.warning(
                            f"Aucun texte extrait de "
                            f"{uploaded_file.name}."
                        )
                        continue

                    st.write("📖 Texte extrait")


                    # =====================================
                    # PHASE 2 : CHUNKING
                    # =====================================

                    chunks = split_text_into_chunks(
                        content,
                        chunk_size=500,
                        chunk_overlap=100
                    )

                    st.write(
                        f"✂️ {len(chunks)} chunks créés"
                    )


                    # =====================================
                    # PHASE 3 : EMBEDDINGS
                    # =====================================

                    embeddings = create_embeddings(chunks)

                    st.write(
                        f"🧠 {len(embeddings)} embeddings générés "
                        f"(dimension {embeddings.shape[1]})"
                    )


                    # =====================================
                    # PHASE 4 : VECTOR STORE - CHROMADB
                    # =====================================

                    add_chunks_to_store(
                        chunks,
                        embeddings,
                        uploaded_file.name
                    )

                    st.write(
                        "🗄️ Ajouté à ChromaDB "
                        f"(total : {get_document_count()} chunks)"
                    )


                    # Marquer ce fichier comme traité
                    st.session_state.processed_files.add(
                        uploaded_file.name
                    )

                    status.update(
                        label=f"✅ {uploaded_file.name} traité",
                        state="complete"
                    )

                except Exception as e:

                    st.error(
                        f"Erreur lors du traitement de "
                        f"{uploaded_file.name} : {e}"
                    )

                    st.exception(e)


# =====================================
# LISTE DES DOCUMENTS DISPONIBLES
# =====================================

if st.session_state.processed_files:

    st.success(
        "📚 Documents disponibles : "
        + ", ".join(st.session_state.processed_files)
    )


# =====================================
# HISTORIQUE DE CONVERSATION
# =====================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            st.write("### 📚 Sources utilisées")

            for i, source in enumerate(message["sources"]):

                with st.expander(
                    f"Source {i + 1} — "
                    f"{source['metadata']['source']} "
                    f"(distance : {source['distance']:.4f})"
                ):

                    st.write(source["document"])


# =====================================
# PHASE 5 + 6 + 7 : CHAT
# =====================================

if st.session_state.processed_files:

    question = st.chat_input(
        "Posez une question sur vos documents..."
    )

    if question:

        with st.chat_message("user"):
            st.write(question)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        try:

            # =====================================
            # PHASE 5 : RETRIEVAL
            # =====================================

            with st.spinner(
                "🔎 Recherche des informations pertinentes..."
            ):

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

            with st.spinner("🤖 Génération de la réponse..."):

                answer = generate_answer(question, documents)


            # =====================================
            # PRÉPARER LES SOURCES
            # =====================================

            sources = []

            for i in range(len(documents)):

                sources.append(
                    {
                        "document": documents[i],
                        "metadata": metadatas[i],
                        "distance": distances[i]
                    }
                )


            # =====================================
            # AFFICHER RÉPONSE
            # =====================================

            with st.chat_message("assistant"):

                st.write(answer)

                st.write("### 📚 Sources utilisées")

                for i, source in enumerate(sources):

                    with st.expander(
                        f"Source {i + 1} — "
                        f"{source['metadata']['source']} "
                        f"(distance : {source['distance']:.4f})"
                    ):

                        st.write(source["document"])


            # =====================================
            # SAUVEGARDER LA RÉPONSE
            # =====================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )

        except Exception as e:

            st.error(f"Erreur lors de la recherche : {e}")
            st.exception(e)

else:

    st.info(
        "👆 Téléversez au moins un document "
        "pour commencer la conversation."
    )