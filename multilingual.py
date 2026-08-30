import re

from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "facebook/nllb-200-distilled-600M"

DetectorFactory.seed = 0


# ============================================================
# SUPPORTED LANGUAGES
# ============================================================

LANGUAGE_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "es": "spa_Latn",
}


LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "es": "Spanish",
}


# ============================================================
# LOAD TRANSLATION MODEL
# ============================================================

print("🌍 Loading multilingual translation model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

print("✅ Multilingual translation model loaded.")


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text):
    """
    Detect the dominant language of a text.

    Returns:
        language code such as en, hi, bn, es
    """

    text = text.strip()

    if not text:
        return "en"

    try:
        detected = detect(text)

    except Exception:
        detected = "en"

    if detected not in LANGUAGE_CODES:
        return "en"

    return detected


# ============================================================
# MIXED LANGUAGE DETECTION
# ============================================================

def detect_mixed_languages(text):
    """
    Detect languages sentence-by-sentence.

    This allows inputs such as:

        "Explain machine learning मुझे समझ नहीं आ रहा"

    to contain multiple languages.
    """

    sentences = re.split(
        r"(?<=[.!?।])\s+",
        text.strip()
    )

    detected_languages = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        language = detect_language(sentence)

        if language not in detected_languages:
            detected_languages.append(language)

    if not detected_languages:
        detected_languages.append("en")

    return detected_languages


# ============================================================
# TRANSLATION
# ============================================================

def translate_text(
    text,
    source_language,
    target_language
):
    """
    Translate text using NLLB-200.
    """

    if not text:
        return text

    if source_language == target_language:
        return text

    source_code = LANGUAGE_CODES.get(
        source_language,
        "eng_Latn"
    )

    target_code = LANGUAGE_CODES.get(
        target_language,
        "eng_Latn"
    )

    tokenizer.src_lang = source_code

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    translated_tokens = translation_model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(
            target_code
        ),
        max_length=512
    )

    translated_text = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    return translated_text


# ============================================================
# MIXED-LANGUAGE NORMALIZATION
# ============================================================

def normalize_to_english(text):
    """
    Convert multilingual user input into English.

    For supported languages, translation is performed.

    Mixed-language input is processed sentence-by-sentence.
    """

    languages = detect_mixed_languages(text)

    # --------------------------------------------------------
    # Pure English
    # --------------------------------------------------------

    if len(languages) == 1 and languages[0] == "en":

        return text, "en", languages

    # --------------------------------------------------------
    # Mixed-language input
    # --------------------------------------------------------

    sentences = re.split(
        r"(?<=[.!?।])\s+",
        text.strip()
    )

    translated_sentences = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        language = detect_language(sentence)

        if language == "en":

            translated_sentences.append(
                sentence
            )

        else:

            translated_sentence = translate_text(
                sentence,
                language,
                "en"
            )

            translated_sentences.append(
                translated_sentence
            )

    normalized_text = " ".join(
        translated_sentences
    )

    return (
        normalized_text,
        languages[0],
        languages
    )


# ============================================================
# TRANSLATE RESPONSE
# ============================================================

def translate_response(
    text,
    target_language
):
    """
    Translate the assistant's English response
    back to the user's preferred language.
    """

    if target_language == "en":
        return text

    return translate_text(
        text,
        "en",
        target_language
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("🌍 MULTILINGUAL SYSTEM TEST")
    print("=" * 60)

    test_messages = [
        "What is machine learning?",
        "मशीन लर्निंग क्या है?",
        "মেশিন লার্নিং কি?",
        "¿Qué es el aprendizaje automático?",
        "What is machine learning? मुझे समझाओ।"
    ]

    for message in test_messages:

        print()
        print(f"User: {message}")

        language = detect_language(
            message
        )

        languages = detect_mixed_languages(
            message
        )

        print(
            f"Detected language: "
            f"{LANGUAGE_NAMES.get(language, language)}"
        )

        print(
            f"Languages detected: "
            f"{[LANGUAGE_NAMES.get(x, x) for x in languages]}"
        )

        english, _, _ = normalize_to_english(
            message
        )

        print(
            f"English meaning: {english}"
        )
