from transformers import pipeline


# ==========================================
# LOAD SENTIMENT MODEL
# ==========================================

print("🧠 Loading sentiment analysis model...")

sentiment_pipeline = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None
)


# ==========================================
# ANALYZE SENTIMENT
# ==========================================

def analyze_sentiment(text):

    if not text or not text.strip():

        return {
            "sentiment": "neutral",
            "confidence": 0.0
        }

    results = sentiment_pipeline(
        text,
        truncation=True
    )[0]

    best_result = max(
        results,
        key=lambda x: x["score"]
    )

    label = best_result["label"].lower()

    if "positive" in label:
        sentiment = "positive"

    elif "negative" in label:
        sentiment = "negative"

    else:
        sentiment = "neutral"

    return {
        "sentiment": sentiment,
        "confidence": best_result["score"]
    }


# ==========================================
# RESPONSE INSTRUCTION
# ==========================================

def get_sentiment_instruction(sentiment):

    if sentiment == "positive":

        return (
            "The user appears positive. "
            "Respond in a friendly, encouraging, "
            "and positive tone."
        )

    elif sentiment == "negative":

        return (
            "The user appears frustrated or negative. "
            "Respond empathetically and calmly. "
            "Acknowledge their concern and avoid "
            "sounding dismissive."
        )

    else:

        return (
            "The user's sentiment appears neutral. "
            "Respond clearly, professionally, "
            "and directly."
        )


# ==========================================
# TEST
# ==========================================

def main():

    print("=" * 60)
    print("😊 SENTIMENT ANALYSIS TEST")
    print("=" * 60)

    print()
    print("Type 'exit' to quit.")

    while True:

        text = input(
            "\nUser message: "
        ).strip()

        if text.lower() == "exit":
            break

        result = analyze_sentiment(text)

        print()
        print(
            "Sentiment:",
            result["sentiment"]
        )

        print(
            "Confidence:",
            f"{result['confidence']:.2%}"
        )

        print()
        print("Response instruction:")

        print(
            get_sentiment_instruction(
                result["sentiment"]
            )
        )


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":
    main()