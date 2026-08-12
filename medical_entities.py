import re

DISEASES = [
    "leukemia",
    "acute lymphoblastic leukemia",
    "acute myeloid leukemia",
    "diabetes",
    "diabetes mellitus",
    "cancer",
    "asthma",
    "pneumonia",
    "covid-19",
    "influenza",
    "hypertension",
    "heart disease",
    "arthritis",
    "alzheimer's disease",
    "parkinson's disease",
]

SYMPTOMS = [
    "fever",
    "fatigue",
    "feeling tired",
    "weakness",
    "headache",
    "cough",
    "shortness of breath",
    "pain",
    "weight loss",
    "nausea",
    "vomiting",
    "bruising",
    "bleeding",
    "swelling",
    "dizziness",
    "loss of appetite",
]

TREATMENTS = [
    "chemotherapy",
    "radiation therapy",
    "radiotherapy",
    "surgery",
    "immunotherapy",
    "transplant",
    "bone marrow transplant",
    "medication",
    "antibiotics",
]

# Question/medical intents
INTENT_PATTERNS = {
    "symptoms": [
        "symptom",
        "symptoms",
        "sign",
        "signs",
        "feel",
        "feeling",
        "warning sign",
        "warning signs",
    ],

    "treatment": [
        "treatment",
        "treat",
        "treated",
        "therapy",
        "therapies",
        "cure",
        "cured",
        "medicine",
        "medication",
    ],

    "diagnosis": [
        "diagnosis",
        "diagnose",
        "diagnosed",
        "test",
        "tests",
        "testing",
        "detect",
        "detected",
    ],

    "causes": [
        "cause",
        "causes",
        "caused",
        "reason",
        "reasons",
        "why",
    ],

    "prevention": [
        "prevent",
        "prevention",
        "avoid",
        "reduce risk",
    ],
}


def contains_term(text, term):
    """
    Check whether a term exists as a complete phrase.
    """

    pattern = r"\b" + re.escape(term) + r"\b"

    return bool(re.search(pattern, text.lower()))


def find_entities(text):

    text_lower = text.lower()

    entities = {
        "diseases": [],
        "symptoms": [],
        "treatments": [],
        "intents": []
    }

    # ------------------------------
    # Diseases
    # ------------------------------

    for disease in DISEASES:

        if contains_term(text_lower, disease):
            entities["diseases"].append(disease)

    # ------------------------------
    # Specific symptoms
    # ------------------------------

    for symptom in SYMPTOMS:

        if contains_term(text_lower, symptom):
            entities["symptoms"].append(symptom)

    # ------------------------------
    # Specific treatments
    # ------------------------------

    for treatment in TREATMENTS:

        if contains_term(text_lower, treatment):
            entities["treatments"].append(treatment)

    # ------------------------------
    # Medical question intent
    # ------------------------------

    for intent, keywords in INTENT_PATTERNS.items():

        for keyword in keywords:

            if contains_term(text_lower, keyword):

                if intent not in entities["intents"]:
                    entities["intents"].append(intent)

                break

    return entities


def display_entities(entities):

    print("\n🧬 DETECTED MEDICAL INFORMATION")
    print("=" * 50)

    print("Diseases   :", entities["diseases"])
    print("Symptoms   :", entities["symptoms"])
    print("Treatments :", entities["treatments"])
    print("Intents    :", entities["intents"])


if __name__ == "__main__":

    print("=" * 50)
    print("🧬 MEDICAL ENTITY & INTENT RECOGNITION")
    print("=" * 50)

    question = input("\nEnter a medical question: ")

    entities = find_entities(question)

    display_entities(entities)