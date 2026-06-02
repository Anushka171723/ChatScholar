import os
import re

from groq import Groq

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def grade_essay(text):
    if not text:
        return empty_grade()

    groq_result = grade_with_groq(text)
    if groq_result:
        return groq_result

    return grade_locally(text)


def empty_grade():
    return {
        "score": 0,
        "summary": "Paste an essay to get feedback.",
        "strengths": [],
        "weaknesses": [],
        "suggestions": [],
    }


def grade_with_groq(text):
    if not os.getenv("GROQ_API_KEY"):
        return None

    try:
        client = Groq()
        completion = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict but helpful essay grader. Return concise "
                        "feedback with a numeric score out of 100."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Grade this essay. Format the response exactly as:\n"
                        "Score: <number>/100\n"
                        "Summary: <one sentence>\n"
                        "Strengths:\n"
                        "- <point>\n"
                        "- <point>\n"
                        "Weaknesses:\n"
                        "- <point>\n"
                        "- <point>\n"
                        "Suggestions:\n"
                        "- <point>\n"
                        "- <point>\n\n"
                        f"Essay:\n{text[:8000]}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=800,
        )
        content = completion.choices[0].message.content.strip()
        return parse_groq_grade(content)
    except Exception as error:
        result = grade_locally(text)
        result["suggestions"].append(f"Groq grading was unavailable: {error}")
        return result


def parse_groq_grade(content):
    score_match = re.search(r"Score:\s*(\d{1,3})\s*/\s*100", content, re.I)
    summary_match = re.search(r"Summary:\s*(.+)", content, re.I)
    return {
        "score": min(int(score_match.group(1)), 100) if score_match else 75,
        "summary": summary_match.group(1).strip()
        if summary_match
        else "Groq generated essay feedback.",
        "strengths": section_items(content, "Strengths", "Weaknesses"),
        "weaknesses": section_items(content, "Weaknesses", "Suggestions"),
        "suggestions": section_items(content, "Suggestions", None),
    }


def section_items(content, heading, next_heading):
    pattern = rf"{heading}:\s*(.*)"
    if next_heading:
        pattern = rf"{heading}:\s*(.*?)(?:{next_heading}:|$)"

    match = re.search(pattern, content, re.I | re.S)
    if not match:
        return []

    return [
        line.strip(" -")
        for line in match.group(1).splitlines()
        if line.strip().startswith("-")
    ]


def grade_locally(text):
    words = re.findall(r"\b[\w']+\b", text)
    sentences = re.findall(r"[^.!?]+[.!?]", text)
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    avg_sentence_length = word_count / sentence_count
    score = 60
    strengths = []
    weaknesses = []
    suggestions = []

    if word_count >= 250:
        score += 15
        strengths.append("Good length for developing an argument.")
    else:
        weaknesses.append("The essay is short for a developed response.")
        suggestions.append("Add more evidence, examples, and explanation.")

    if 12 <= avg_sentence_length <= 24:
        score += 10
        strengths.append("Sentence length is easy to read.")
    else:
        weaknesses.append("Sentence rhythm could be smoother.")
        suggestions.append("Vary sentence length and break up long sentences.")

    if has_clear_structure(text):
        score += 10
        strengths.append("The essay shows structure and transitions.")
    else:
        weaknesses.append("Transitions and paragraph structure need work.")
        suggestions.append("Use clear introduction, body, and conclusion sections.")

    if text.count(",") + text.count(";") >= 4:
        score += 5
        strengths.append("Punctuation helps shape the ideas.")
    else:
        weaknesses.append("Punctuation use is limited.")
        suggestions.append("Use commas and punctuation to separate ideas clearly.")

    return {
        "score": min(score, 100),
        "summary": f"{word_count} words, about {sentence_count} sentences.",
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
    }


def has_clear_structure(text):
    markers = ["first", "second", "finally", "therefore", "however", "in conclusion"]
    lowered = text.lower()
    return any(marker in lowered for marker in markers) or len(text.split("\n\n")) >= 3
