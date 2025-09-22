"""Utility functions for generating Blooket-style question sets with OpenAI."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from openai import OpenAI

MAX_TIME_LIMIT = 300
DEFAULT_TEMPLATE_PATHS: Dict[str, Path] = {
    "blooket": Path("data/blooket_template.csv"),
    "gimkit": Path("data/gimkit_template.csv"),
}


@dataclass
class QuestionItem:
    """Container for a single Blooket style question."""

    prompt: str
    answers: List[str]
    correct_answers: List[int]
    time_limit: int = 20
    explanation: str | None = None

    def normalized(self) -> "QuestionItem":
        answers = [ans.strip() for ans in self.answers if ans and ans.strip()]
        if not answers:
            raise ValueError("No answer choices provided.")
        if len(answers) > 4:
            answers = answers[:4]
        while len(answers) < 4:
            answers.append("")

        normalized_correct: List[int] = []
        for idx in self.correct_answers:
            if isinstance(idx, str):
                if not idx.strip():
                    continue
                idx = int(idx)
            if idx < 1 or idx > len(answers):
                raise ValueError(
                    f"Correct answer index {idx} out of bounds for answers: {answers}"
                )
            normalized_correct.append(idx)

        if not normalized_correct:
            raise ValueError("At least one correct answer index is required.")

        time_limit = max(1, min(int(self.time_limit or 20), MAX_TIME_LIMIT))

        return QuestionItem(
            prompt=self.prompt.strip(),
            answers=answers,
            correct_answers=normalized_correct,
            time_limit=time_limit,
            explanation=self.explanation,
        )


def build_prompt(
    grade: str,
    subject: str,
    assessment_goal: str,
    keywords: Sequence[str],
    num_questions: int,
) -> str:
    keyword_text = ", ".join(str(k).strip() for k in keywords if str(k).strip())
    keyword_block = keyword_text if keyword_text else "None provided"
    return (
        "You are an assistant that helps South Korean teachers create multiple-choice quizzes for Blooket.\n"
        f"Create {num_questions} multiple-choice questions.\n"
        f"Grade level: {grade}\n"
        f"Subject: {subject}\n"
        f"Assessment focus: {assessment_goal}\n"
        f"Key topics: {keyword_block}\n\n"
        "All question stems, answer choices, and explanations must be written in Korean.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        "  \"questions\": [\n"
        "    {\n"
        "      \"prompt\": \"Question text in Korean\",\n"
        "      \"answers\": [\"Option1\", \"Option2\", \"Option3\", \"Option4\"],\n"
        "      \"correct_answers\": [1],  # 1-based indexes, allow multiple correct\n"
        "      \"time_limit\": 20,        # seconds, optional\n"
        "      \"explanation\": \"Rationale in Korean (optional)\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Questions should be concise (two sentences or fewer) and age-appropriate."
    )


def ensure_api_key(api_key: str | None = None) -> str:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set the OPENAI_API_KEY environment variable or add openai_api_key to Streamlit secrets."
        )
    return api_key


def call_openai(
    api_key: str,
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_output_tokens: int = 2000,
) -> str:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You produce only valid JSON responses for Korean multiple-choice quizzes.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_output_tokens,
    )
    return response.choices[0].message.content


def parse_questions(raw_json: str) -> List[QuestionItem]:
    payload = json.loads(raw_json)
    questions = payload.get("questions")
    if not isinstance(questions, Iterable):
        raise ValueError("JSON response does not contain a 'questions' array.")

    parsed: List[QuestionItem] = []
    for idx, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {idx} is not an object: {item}")
        prompt = item.get("prompt", "").strip()
        answers = item.get("answers", [])
        correct = item.get("correct_answers", [])
        time_limit = item.get("time_limit", 20)
        explanation = item.get("explanation")

        if not prompt:
            raise ValueError(f"Question {idx} is missing the prompt text.")
        if not isinstance(answers, Sequence):
            raise ValueError(f"Question {idx} has an invalid 'answers' field.")
        if isinstance(correct, (int, str)):
            correct_list = [correct]
        elif isinstance(correct, Sequence):
            correct_list = list(correct)
        else:
            raise ValueError(f"Question {idx} has an invalid 'correct_answers' field.")

        question = QuestionItem(
            prompt=prompt,
            answers=list(answers),
            correct_answers=[int(c) for c in correct_list],
            time_limit=int(time_limit) if time_limit else 20,
            explanation=explanation,
        ).normalized()
        parsed.append(question)

    if not parsed:
        raise ValueError("No questions were generated.")
    return parsed


def generate_question_set(
    grade: str,
    subject: str,
    assessment_goal: str,
    keywords: Sequence[str],
    num_questions: int,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    api_key: str | None = None,
) -> List[QuestionItem]:
    api_key = ensure_api_key(api_key)
    prompt = build_prompt(grade, subject, assessment_goal, keywords, num_questions)
    raw_json = call_openai(
        api_key=api_key,
        prompt=prompt,
        model=model,
        temperature=temperature,
    )
    return parse_questions(raw_json)


def resolve_template_path(platform: str, override_path: str | os.PathLike[str] | None = None) -> Path:
    if override_path:
        return Path(override_path)
    try:
        return DEFAULT_TEMPLATE_PATHS[platform]
    except KeyError as exc:
        raise ValueError(f"Unsupported platform: {platform}") from exc


def load_template_columns(platform: str, template_path: str | os.PathLike[str] | None = None) -> List[str]:
    """Return the CSV header row for the requested platform."""

    path = resolve_template_path(platform, template_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Blooket template not found: {path}. Place the CSV template in the project folder or upload your own."
        )

    with path.open("r", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])

    header = [column.strip() for column in header if column.strip()]
    if not header:
        raise ValueError("Template CSV header row is empty.")
    return header


def build_question_csv(
    questions: Sequence[QuestionItem],
    template_columns: Sequence[str],
    platform: str,
) -> bytes:
    """Build a CSV payload that matches the chosen platform format."""

    if not questions:
        raise ValueError("No questions to export.")
    if not template_columns:
        raise ValueError("Template columns are required.")

    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(template_columns)

    for idx, question in enumerate(questions, start=1):
        normalized_answers = list(question.answers)

        if platform == "blooket":
            column_values = {
                "question #": idx,
                "question text": question.prompt,
                "answer 1": normalized_answers[0] if len(normalized_answers) > 0 else "",
                "answer 2": normalized_answers[1] if len(normalized_answers) > 1 else "",
                "answer 3": normalized_answers[2] if len(normalized_answers) > 2 else "",
                "answer 4": normalized_answers[3] if len(normalized_answers) > 3 else "",
                "time limit (sec)": question.time_limit,
                "correct answer(s)": ",".join(str(i) for i in question.correct_answers),
            }
        elif platform == "gimkit":
            correct_idx = question.correct_answers[0] if question.correct_answers else 1
            correct_idx = max(1, min(correct_idx, len(normalized_answers)))
            correct_answer = normalized_answers[correct_idx - 1]
            incorrect = [ans for i, ans in enumerate(normalized_answers, start=1) if i != correct_idx and ans]
            while len(incorrect) < 3:
                incorrect.append("")
            column_values = {
                "question": question.prompt,
                "correct answer": correct_answer,
                "incorrect answer 1": incorrect[0] if len(incorrect) > 0 else "",
                "incorrect answer 2 (optional)": incorrect[1] if len(incorrect) > 1 else "",
                "incorrect answer 3 (optional)": incorrect[2] if len(incorrect) > 2 else "",
            }
        else:
            raise ValueError(f"Unsupported platform: {platform}")

        row = []
        for column in template_columns:
            value = column_values.get(column.strip().lower(), "")
            row.append(value)
        writer.writerow(row)

    return output.getvalue().encode("utf-8-sig")


__all__ = [
    "QuestionItem",
    "generate_question_set",
    "build_question_csv",
    "load_template_columns",
    "resolve_template_path",
]
