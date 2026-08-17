import streamlit as st

from arxiv_qa import (
    load_papers,
    create_database,
    generate_research_answer
)


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="ArXiv Research Assistant",
    page_icon="🔬",
    layout="wide"
)


# ==========================================
# HEADER
# ==========================================

st.title("🔬 ArXiv Research Q&A Assistant")

st.write(
    "Ask questions about Computer Science research "
    "using a knowledge base of arXiv papers."
)

st.info(
    "📚 This assistant retrieves relevant research papers "
    "and generates answers grounded in the retrieved literature."
)


# ==========================================
# SESSION STATE
# ==========================================

if "research_history" not in st.session_state:
    st.session_state.research_history = []


# ==========================================
# LOAD RESEARCH DATABASE
# ==========================================

@st.cache_resource
def load_research_database():

    papers = load_papers()

    collection = create_database(
        papers
    )

    return collection


# ==========================================
# DATABASE STATUS
# ==========================================

database_ready = False
collection = None

with st.spinner(
    "📚 Loading research knowledge base..."
):

    try:

        collection = load_research_database()

        database_ready = True

    except Exception as e:

        st.error(
            f"❌ Failed to load research database: {e}"
        )


# ==========================================
# HELPER FUNCTION
# ==========================================

def format_sources(results):
    """
    Convert the raw ChromaDB result into
    a list of dictionaries that Streamlit
    can display easily.
    """

    if not results:
        return []

    documents = results.get(
        "documents",
        [[]]
    )

    metadatas = results.get(
        "metadatas",
        [[]]
    )

    distances = results.get(
        "distances",
        [[]]
    )

    if not documents:
        return []

    documents = documents[0] if documents else []
    metadatas = metadatas[0] if metadatas else []
    distances = distances[0] if distances else []

    formatted = []

    for index in range(len(metadatas)):

        metadata = metadatas[index]

        distance = None

        if index < len(distances):
            distance = distances[index]

        document = ""

        if index < len(documents):
            document = documents[index]

        # Extract abstract from document
        abstract = ""

        if "Abstract:" in document:

            abstract = document.split(
                "Abstract:",
                1
            )[1].strip()

        formatted.append(
            {
                "title": metadata.get(
                    "title",
                    "Research Paper"
                ),

                "authors": metadata.get(
                    "authors",
                    ""
                ),

                "categories": metadata.get(
                    "categories",
                    ""
                ),

                "published": metadata.get(
                    "published",
                    ""
                ),

                "arxiv_id": metadata.get(
                    "arxiv_id",
                    ""
                ),

                "abstract": abstract,

                "distance": distance
            }
        )

    return formatted


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header(
        "🔬 ArXiv Research Assistant"
    )

    st.write(
        "Research assistant powered by:"
    )

    st.write(
        "• arXiv research papers"
    )

    st.write(
        "• ChromaDB"
    )

    st.write(
        "• Sentence embeddings"
    )

    st.write(
        "• Gemini"
    )

    st.divider()

    if database_ready:

        st.success(
            f"📚 Papers available: "
            f"{collection.count()}"
        )

    else:

        st.error(
            "Research database unavailable"
        )

    st.divider()

    st.subheader(
        "💡 Example questions"
    )

    st.write(
        "• How are transformers used in computer vision?"
    )

    st.write(
        "• What are the recent applications of large language models?"
    )

    st.write(
        "• How is reinforcement learning used in robotics?"
    )

    st.write(
        "• What are the challenges of deploying AI on edge devices?"
    )

    st.divider()

    st.caption(
        "Research answers are generated from "
        "the retrieved arXiv papers."
    )


# ==========================================
# DISPLAY PREVIOUS RESULTS
# ==========================================

for item in st.session_state.research_history:

    with st.chat_message("user"):

        st.markdown(
            item["question"]
        )

    with st.chat_message("assistant"):

        st.markdown(
            item["answer"]
        )

        sources = item.get(
            "results",
            []
        )

        if sources:

            with st.expander(
                "📚 View research sources"
            ):

                for i, result in enumerate(
                    sources,
                    start=1
                ):

                    st.markdown(
                        f"### {i}. "
                        f"{result.get('title', 'Research Paper')}"
                    )

                    if result.get("authors"):

                        st.write(
                            "👨‍🔬 Authors:",
                            result["authors"]
                        )

                    if result.get("categories"):

                        st.write(
                            "🏷️ Categories:",
                            result["categories"]
                        )

                    if result.get("published"):

                        st.write(
                            "📅 Published:",
                            result["published"]
                        )

                    if result.get("arxiv_id"):

                        st.write(
                            "🔗 arXiv:",
                            result["arxiv_id"]
                        )

                    if result.get("distance") is not None:

                        st.write(
                            "📏 Retrieval distance:",
                            result["distance"]
                        )

                    if result.get("abstract"):

                        with st.expander(
                            "📄 Abstract"
                        ):

                            st.write(
                                result["abstract"]
                            )

                    st.divider()


# ==========================================
# USER INPUT
# ==========================================

question = st.chat_input(
    "Ask a research question..."
)


# ==========================================
# PROCESS QUESTION
# ==========================================

if question:

    if not database_ready:

        st.error(
            "❌ The research database could not be loaded."
        )

        st.stop()


    # --------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------

    with st.chat_message("user"):

        st.markdown(
            question
        )


    # --------------------------------------
    # GENERATE ANSWER
    # --------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🧠 Searching research papers and generating answer..."
        ):

            try:

                answer, raw_results = (
                    generate_research_answer(
                        question,
                        collection
                    )
                )


                # ----------------------------------
                # FORMAT SOURCES
                # ----------------------------------

                sources = format_sources(
                    raw_results
                )


                # ----------------------------------
                # DISPLAY ANSWER
                # ----------------------------------

                st.markdown(
                    answer
                )


                # ----------------------------------
                # DISPLAY SOURCES
                # ----------------------------------

                with st.expander(
                    "📚 Research sources used"
                ):

                    if sources:

                        for i, result in enumerate(
                            sources,
                            start=1
                        ):

                            st.markdown(
                                f"### {i}. "
                                f"{result.get('title', 'Research Paper')}"
                            )

                            if result.get("authors"):

                                st.write(
                                    "👨‍🔬 Authors:",
                                    result["authors"]
                                )

                            if result.get("categories"):

                                st.write(
                                    "🏷️ Categories:",
                                    result["categories"]
                                )

                            if result.get("published"):

                                st.write(
                                    "📅 Published:",
                                    result["published"]
                                )

                            if result.get("arxiv_id"):

                                st.write(
                                    "🔗 arXiv:",
                                    result["arxiv_id"]
                                )

                            if result.get("distance") is not None:

                                st.write(
                                    "📏 Retrieval distance:",
                                    result["distance"]
                                )

                            if result.get("abstract"):

                                with st.expander(
                                    "📄 Abstract"
                                ):

                                    st.write(
                                        result["abstract"]
                                    )

                            st.divider()

                    else:

                        st.write(
                            "No research sources were returned."
                        )


                # ----------------------------------
                # SAVE HISTORY
                # ----------------------------------

                st.session_state.research_history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "results": sources
                    }
                )


            except Exception as e:

                st.error(
                    f"❌ Error generating research answer: {e}"
                )