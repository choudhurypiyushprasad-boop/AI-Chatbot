import os
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-3.5-flash"

conversation_history = []


def validate_response(question, answer):
    """
    Validate the generated response before returning it to the user.
    """

    validation_prompt = f"""
You are a response validation system for a multimodal AI assistant.

Review the answer below.

Question:
{question}

Generated answer:
{answer}

Check whether the answer:
1. Directly addresses the question.
2. Is supported by information visible in the image.
3. Clearly distinguishes observations from assumptions.
4. Avoids inventing names, locations, dates, identities, or other facts.
5. Acknowledges uncertainty when the image does not provide enough evidence.

Return ONLY one of these formats:

VALID
<short reason>

or

INVALID
<short reason>
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=validation_prompt
        )

        result = response.text.strip()

        if result.upper().startswith("VALID"):
            return answer

        # If validation fails, return a safe response.
        return (
            "I couldn't verify that the generated answer is fully "
            "supported by the available image evidence."
        )

    except Exception:
        # Fail safely if the validation service is unavailable.
        return answer


def analyze_image(image_path, question):
    """
    Analyze an image while maintaining conversation context
    and validating the generated response.
    """

    try:
        image = Image.open(image_path)

        conversation_history.append(f"User: {question}")

        history = "\n".join(conversation_history)

        prompt = f"""
You are a multimodal AI assistant that analyzes images
and maintains context across multiple questions.

Conversation history:
{history}

Current question:
{question}

Rules:
1. Answer using information supported by the image.
2. Clearly distinguish visible evidence from inference.
3. If information cannot be determined from the image, say so.
4. Never invent names, locations, dates, identities, or other facts.
5. Resolve references such as "it", "they", "that", and "the person"
   using the previous conversation.
6. Keep the answer clear and concise.
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image]
        )

        answer = response.text

        # Validate generated response
        validated_answer = validate_response(question, answer)

        conversation_history.append(
            f"Assistant: {validated_answer}"
        )

        return validated_answer

    except FileNotFoundError:
        return "Error: The specified image file was not found."

    except Exception as e:
        return f"Error while analyzing image: {e}"


def clear_conversation():
    """Clear conversation history."""
    conversation_history.clear()


if __name__ == "__main__":

    print("=" * 60)
    print("🖼️ MULTIMODAL AI ASSISTANT")
    print("=" * 60)

    image_path = input("\nEnter image path: ")

    while True:

        question = input("\nYou: ")

        if question.lower() == "exit":
            break

        if question.lower() == "clear":
            clear_conversation()
            print("🧹 Conversation cleared.")
            continue

        answer = analyze_image(image_path, question)

        print("\n🤖 Assistant:")
        print(answer)