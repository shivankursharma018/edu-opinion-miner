"""
STEP 2: ASPECT EXTRACTION
=========================
Simple keyword-based approach.
For each review, we detect which aspects are mentioned.
This is easy to understand and explain in viva.
"""

import pandas as pd

# ─────────────────────────────────────────────
# KEYWORD DICTIONARY (the core of aspect extraction)
# ─────────────────────────────────────────────
ASPECT_KEYWORDS = {
    "course_content": [
        "syllabus",
        "curriculum",
        "course",
        "content",
        "topic",
        "subject",
        "module",
        "material taught",
        "lecture content",
        "course material",
        "study plan",
        "course structure",
        "theory",
    ],
    "teaching_quality": [
        "teacher",
        "professor",
        "faculty",
        "instructor",
        "teaching",
        "lecture",
        "explain",
        "class",
        "staff",
        "mentor",
        "trainer",
        "educator",
        "teaches",
        "taught",
        "professor",
    ],
    "assessment": [
        "exam",
        "test",
        "assignment",
        "quiz",
        "grade",
        "grading",
        "marks",
        "evaluation",
        "assessment",
        "project",
        "deadline",
        "score",
        "viva",
        "practical exam",
        "paper",
    ],
    "resources": [
        "library",
        "book",
        "material",
        "resource",
        "journal",
        "reference",
        "notes",
        "handout",
        "e-resource",
        "database",
        "study material",
        "online resource",
        "pdf",
    ],
    "infrastructure": [
        "lab",
        "classroom",
        "wifi",
        "internet",
        "equipment",
        "facility",
        "campus",
        "projector",
        "computer",
        "infrastructure",
        "building",
        "canteen",
        "hostel",
        "smart board",
        "machine",
    ],
}


def extract_aspects(text):
    """
    Check which aspects are mentioned in the text.
    Returns a dict: {aspect: True/False}

    Simple approach: if any keyword from the aspect's list
    appears in the text → aspect is mentioned.
    """
    text_lower = text.lower()
    mentioned = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        mentioned[aspect] = any(kw in text_lower for kw in keywords)
    return mentioned


def demonstrate_extraction():
    """Show how aspect extraction works on examples."""
    examples = [
        "Great course content but difficult exams and poor lab equipment",
        "The professor explains well and library resources are excellent",
        "Syllabus is outdated and grading is very strict",
        "Campus wifi is very slow and the teacher is not helpful",
    ]

    print("=" * 60)
    print("ASPECT EXTRACTION DEMO")
    print("=" * 60)
    for text in examples:
        detected = extract_aspects(text)
        mentioned = [asp for asp, found in detected.items() if found]
        print(f'\nText: "{text}"')
        print(f"Aspects found: {mentioned}")


def add_aspect_flags(df):
    """Add detected columns to dataset."""
    for aspect in ASPECT_KEYWORDS:
        df[f"detected_{aspect}"] = df["text"].apply(
            lambda t: extract_aspects(t)[aspect]
        )
    return df


if __name__ == "__main__":
    demonstrate_extraction()

    # Also apply to our dataset
    df = pd.read_csv("data/dataset.csv")
    df = add_aspect_flags(df)

    print("\n\n" + "=" * 60)
    print("EXTRACTION ACCURACY CHECK")
    print("=" * 60)

    for aspect in ASPECT_KEYWORDS:
        # True positives: label not 'none' AND detected
        labeled = df[df[aspect] != "none"]
        if len(labeled) > 0:
            detected_correctly = labeled[labeled[f"detected_{aspect}"] == True]
            acc = len(detected_correctly) / len(labeled) * 100
            print(
                f"{aspect:25s}: {acc:.0f}% recall ({len(detected_correctly)}/{len(labeled)} correctly detected)"
            )

    df.to_csv("data/dataset_with_aspects.csv", index=False)
    print("\n✅ Saved to data/dataset_with_aspects.csv")
