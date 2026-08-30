from transformers import pipeline


print("=" * 60)
print("📊 TASK 5 - SENTIMENT ANALYSIS EVALUATION")
print("=" * 60)


model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)


test_cases = [

    ("I absolutely love this chatbot!", "positive"),

    ("This is amazing and really helpful.", "positive"),

    ("Thank you, this solved my problem.", "positive"),

    ("What is machine learning?", "neutral"),

    ("Explain how transformers work.", "neutral"),

    ("Can you tell me about RAG?", "neutral"),

    ("This chatbot is useless.", "negative"),

    ("You keep giving me wrong answers.", "negative"),

    ("I'm frustrated because this isn't working.", "negative"),

]


correct = 0


for text, expected in test_cases:

    result = model(
        text,
        truncation=True
    )[0]

    predicted = result["label"].lower()

    confidence = result["score"]

    is_correct = predicted == expected

    if is_correct:
        correct += 1

    status = "✅" if is_correct else "❌"

    print()
    print(status, text)
    print(
        f"Expected : {expected}"
    )
    print(
        f"Predicted: {predicted}"
    )
    print(
        f"Confidence: {confidence * 100:.2f}%"
    )


accuracy = (
    correct / len(test_cases)
) * 100


print()
print("=" * 60)
print("📈 EVALUATION RESULTS")
print("=" * 60)

print(
    f"Total test cases : {len(test_cases)}"
)

print(
    f"Correct          : {correct}"
)

print(
    f"Incorrect        : "
    f"{len(test_cases) - correct}"
)

print(
    f"Accuracy         : {accuracy:.2f}%"
)

print("=" * 60)