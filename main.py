import streamlit as st

from src.loaders.document_loader import extract_text
from src.chunking.text_splitter import split_text_into_chunks
from src.embeddings.embedding_service import create_embeddings

from src.vectorstore.vector_store import (
    add_chunks_to_store,
    get_document_count,
    get_sources
)

from src.retrieval.retriever import search_similar_chunks
from src.llm.groq_client import generate_answer


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocuChat AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .document-card {
        padding: 10px;
        border-radius: 8px;
        background-color: #f5f7fa;
        margin-bottom: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "processed_files" not in st.session_state:

    st.session_state.processed_files = set(
        get_sources()
    )


if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 📚 DocuChat AI")

    st.caption(
        "Chat with your own documents using AI."
    )

    st.divider()

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    st.markdown("### 📄 Documents")

    uploaded_files = st.file_uploader(
        "Ajouter des documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    # --------------------------------------------------------
    # TRAITEMENT DES DOCUMENTS
    # --------------------------------------------------------

    if uploaded_files:

        for uploaded_file in uploaded_files:

            if (
                uploaded_file.name
                not in st.session_state.processed_files
            ):

                with st.spinner(
                    f"Indexation de {uploaded_file.name}..."
                ):

                    try:

                        # ==============================
                        # EXTRACTION
                        # ==============================

                        content = extract_text(
                            uploaded_file
                        )

                        if not content:

                            st.warning(
                                f"Impossible de lire "
                                f"{uploaded_file.name}"
                            )

                            continue

                        # ==============================
                        # CHUNKING
                        # ==============================

                        chunks = split_text_into_chunks(
                            content,
                            chunk_size=500,
                            chunk_overlap=100
                        )

                        # ==============================
                        # EMBEDDINGS
                        # ==============================

                        embeddings = create_embeddings(
                            chunks
                        )

                        # ==============================
                        # CHROMADB
                        # ==============================

                        add_chunks_to_store(
                            chunks,
                            embeddings,
                            uploaded_file.name
                        )

                        st.session_state.processed_files.add(
                            uploaded_file.name
                        )

                    except Exception as e:

                        st.error(
                            f"Erreur lors du traitement de "
                            f"{uploaded_file.name}"
                        )

    # --------------------------------------------------------
    # DOCUMENTS DISPONIBLES
    # --------------------------------------------------------

    st.divider()

    if st.session_state.processed_files:

        st.markdown("### 📚 Vos documents")

        for filename in sorted(
            st.session_state.processed_files
        ):

            st.markdown(
                f"""
                <div class="document-card">
                    📄 {filename}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.caption(
            f"{len(st.session_state.processed_files)} "
            f"document(s)"
        )

    else:

        st.info(
            "Aucun document ajouté."
        )

    # --------------------------------------------------------
    # NOUVEAU CHAT
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "🗑️ Effacer la conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# HEADER PRINCIPAL
# ============================================================

st.markdown(
    '<div class="main-title">📚 DocuChat AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Posez des questions à vos documents avec l’IA.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# ÉTAT VIDE
# ============================================================

if not st.session_state.processed_files:

    st.info(
        "👈 Commencez par ajouter un ou plusieurs "
        "documents depuis la barre latérale."
    )


# ============================================================
# HISTORIQUE DU CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT
# ============================================================

if st.session_state.processed_files:

    question = st.chat_input(
        "Posez une question sur vos documents..."
    )

    if question:

        # ----------------------------------------------------
        # MESSAGE UTILISATEUR
        # ----------------------------------------------------

        with st.chat_message("user"):

            st.markdown(question)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        try:

            # ------------------------------------------------
            # RETRIEVAL
            # ------------------------------------------------

            with st.spinner(
                "Recherche dans vos documents..."
            ):

                results = search_similar_chunks(
                    question,
                    k=60
                )

            documents = results[
                "documents"
            ][0]

            # ------------------------------------------------
            # GENERATION
            # ------------------------------------------------

            with st.spinner(
                "Génération de la réponse..."
            ):

                answer = generate_answer(
                    question,
                    documents
                )

            # ------------------------------------------------
            # MESSAGE ASSISTANT
            # ------------------------------------------------

            with st.chat_message(
                "assistant"
            ):

                st.markdown(answer)

            # ------------------------------------------------
            # HISTORIQUE
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            st.error(
                f"Une erreur est survenue : {e}"
            )