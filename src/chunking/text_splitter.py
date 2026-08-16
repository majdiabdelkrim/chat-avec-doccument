from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_into_chunks(
    text,
    chunk_size=1000,
    chunk_overlap=200
):
    """
    Découpe un texte en plusieurs chunks.

    Args:
        text: Texte complet du document.
        chunk_size: Taille maximale approximative d'un chunk.
        chunk_overlap: Nombre de caractères partagés entre deux chunks.

    Returns:
        Liste des chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    return chunks