import os

from dotenv import load_dotenv
from groq import Groq


# =====================================
# CONFIGURATION GROQ
# =====================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY n'est pas définie dans le fichier .env"
    )

client = Groq(
    api_key=GROQ_API_KEY
)


MODEL_NAME = "llama-3.3-70b-versatile"


# =====================================
# PROMPT SYSTÈME
# =====================================

SYSTEM_PROMPT = """
Tu es un assistant qui répond uniquement à partir du contexte fourni.

N'utilise aucune connaissance extérieure et n'invente aucune information.
Si la réponse ne peut pas être déduite du contexte, réponds exactement :
"Je ne peux pas répondre à cette question à partir des documents fournis."
"""


# =====================================
# GÉNÉRATION DE LA RÉPONSE
# =====================================

def generate_answer(question, chunks):
    """
    Génère une réponse à partir de la question
    et des chunks pertinents récupérés.
    """

    # Construire le contexte à partir des chunks
    context = "\n\n".join(chunks)

    # Construire le prompt utilisateur
    prompt = f"""
CONTEXTE :
{context}

QUESTION :
{question}
"""

    # Appel à Groq
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    # Retourner uniquement la réponse du LLM
    return completion.choices[0].message.content