import streamlit as st
from multimodal.assistant import analyze_image, clear_conversation

st.set_page_config(
    page_title="Multimodal AI Assistant",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Multimodal AI Assistant")
st.write(
    "Upload an image and ask questions about it. "
    "The assistant uses visual evidence, conversation context, "
    "and response validation."
)

# Image upload
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Save uploaded image temporarily
    image_path = "multimodal/test_images/uploaded_image.jpg"

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display image
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.divider()

    # Question input
    question = st.text_input(
        "Ask a question about the image",
        placeholder="Example: What environment is shown?"
    )

    col1, col2 = st.columns(2)

    with col1:
        analyze_button = st.button(
            "🔍 Analyze Image",
            use_container_width=True
        )

    with col2:
        clear_button = st.button(
            "🧹 Clear Conversation",
            use_container_width=True
        )

    if clear_button:
        clear_conversation()
        st.success("Conversation cleared.")

    if analyze_button:

        if not question.strip():
            st.warning("Please enter a question.")

        else:

            with st.spinner("🧠 Analyzing image..."):

                answer = analyze_image(
                    image_path,
                    question
                )

            st.subheader("🤖 Assistant")
            st.write(answer)

else:
    st.info("👆 Upload an image to begin.")