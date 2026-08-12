import os

from dotenv import load_dotenv
from google import genai

from medical_retriever import retrieve_medical_info
from medical_entities import find_entities


# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file"
    )

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.5-flash"


# ==========================================
# BUILD MEDICAL CONTEXT
# ==========================================

def build_context(results):

    context_parts = []

    for i, result in enumerate(results, 1):

        context_parts.append(
            f"""
SOURCE {i}
Medical Topic: {result['focus']}
Dataset Question: {result['question']}

Medical Information:
{result['document']}
"""
        )

    return "\n".join(context_parts)


# ==========================================
# GENERATE MEDICAL ANSWER
# ==========================================

def generate_medical_answer(question):

    # --------------------------------------
    # Detect entities and intent
    # --------------------------------------

    entities = find_entities(question)

    # --------------------------------------
    # Retrieve relevant MedQuAD information
    # --------------------------------------

    results = retrieve_medical_info(
        question,
        top_k=5
    )

    if not results:
        return (
            "I could not find relevant information "
            "in the MedQuAD knowledge base."
        )

    context = build_context(results)

    # --------------------------------------
    # Medical safety instructions
    # --------------------------------------

    prompt = f"""
You are a medical question-answering assistant.

Your answers must be based primarily on the
provided MedQuAD information.

Do NOT invent medical facts that are not supported
by the retrieved information.

If the retrieved information is insufficient,
clearly say that the available information is
insufficient rather than making up an answer.

Do not diagnose the user.

Do not prescribe medication or give personalized
treatment instructions.

For potentially serious symptoms, encourage the
user to consult a qualified healthcare professional.

Keep the answer clear and easy to understand.

Detected medical information:
Diseases: {entities['diseases']}
Symptoms: {entities['symptoms']}
Treatments: {entities['treatments']}
Question intents: {entities['intents']}

Retrieved MedQuAD information:

{context}

User question:

{question}

Provide a concise, evidence-based answer using
the retrieved information.
"""

    # --------------------------------------
    # Gemini generation
    # --------------------------------------

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text


# ==========================================
# CHATBOT
# ==========================================

def main():

    print("=" * 60)
    print("🏥 MEDQUAD MEDICAL Q&A CHATBOT")
    print("=" * 60)

    print("\n⚠️ Medical information assistant.")
    print("This chatbot does not provide a diagnosis.")
    print("For medical concerns, consult a qualified healthcare professional.")

    while True:

        question = input(
            "\nYou: "
        ).strip()

        if not question:
            continue

        if question.lower() in [
            "exit",
            "quit",
            "bye"
        ]:
            print("\n🏥 Medical chatbot stopped.")
            break

        try:

            answer = generate_medical_answer(
                question
            )

            print("\nBot:")
            print(answer)

        except Exception as e:

            print(
                f"\n❌ Error: {e}"
            )


if __name__ == "__main__":
    main()