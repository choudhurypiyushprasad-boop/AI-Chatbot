import streamlit as st

from medical_qa import generate_medical_answer
from medical_entities import find_entities


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="MedQuAD Medical Assistant",
    page_icon="🏥",
    layout="centered"
)


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# HEADER
# ==========================================

st.title("🏥 MedQuAD Medical Q&A Assistant")

st.write(
    "Ask medical questions and receive answers grounded "
    "in the MedQuAD knowledge base."
)

st.warning(
    "⚠️ This assistant provides medical information only. "
    "It does not diagnose conditions or replace professional "
    "medical advice."
)


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================
# USER INPUT
# ==========================================

question = st.chat_input(
    "Ask a medical question...",
    key="medical_question_input"
)


# ==========================================
# PROCESS QUESTION
# ==========================================

if question:

    # --------------------------------------
    # Display user question
    # --------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # --------------------------------------
    # Generate medical answer
    # --------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching MedQuAD and generating an answer..."
        ):

            try:

                # Generate answer using MedQuAD + Gemini
                answer = generate_medical_answer(question)

                # Display answer
                st.markdown(answer)

                # --------------------------------------
                # Medical entity and intent detection
                # --------------------------------------

                entities = find_entities(question)

                with st.expander(
                    "🧬 Detected medical information"
                ):

                    st.write(
                        "**Diseases:**",
                        entities.get("diseases", [])
                    )

                    st.write(
                        "**Symptoms:**",
                        entities.get("symptoms", [])
                    )

                    st.write(
                        "**Treatments:**",
                        entities.get("treatments", [])
                    )

                    st.write(
                        "**Question intents:**",
                        entities.get("intents", [])
                    )

            except Exception as e:

                answer = (
                    "Sorry, an error occurred while "
                    "processing your question."
                )

                st.error(
                    f"Error details: {e}"
                )

    # --------------------------------------
    # Save assistant response
    # --------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("🏥 MedQuAD")

    st.write(
        "Medical Q&A chatbot powered by:"
    )

    st.write("• MedQuAD dataset")
    st.write("• ChromaDB retrieval")
    st.write("• Sentence embeddings")
    st.write("• Gemini")

    st.divider()

    st.subheader("Example questions")

    st.write("• What are the symptoms of leukemia?")
    st.write("• What causes diabetes?")
    st.write("• What are the treatments for cancer?")
    st.write("• How is leukemia diagnosed?")

    st.divider()

    st.caption(
        "For educational purposes only. "
        "Consult a qualified healthcare professional "
        "for medical concerns."
    )