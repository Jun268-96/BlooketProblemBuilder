from __future__ import annotations

import json
from typing import List, Sequence

from openai import OpenAI

from blooket_generator import QuestionItem, ensure_api_key

REVIEW_MODEL = "gpt-4o-mini"
REVIEW_VERDICTS = {"pass", "fail", "uncertain"}


def build_review_prompt(questions: Sequence[QuestionItem], reference_context: str | None = None) -> str:
    header = (
        "You are an educational content reviewer verifying Korean multiple-choice questions for accuracy.\n"
        "For each item, decide whether the marked correct answers are valid and the explanation is coherent.\n"
        "Use the following verdicts strictly: pass, fail, uncertain.\n"
        "If you identify any issue, describe it briefly in Korean.\n"
    )

    if reference_context and reference_context.strip():
        header += (
            "When available, use the provided reference material as the primary source of truth. "
            "If the reference does not cover a question, make a best-effort judgement but prefer 'uncertain' over guessing.\n"
            "Reference material (Korean):\n"
            f"{reference_context.strip()}\n\n"
        )

    body_lines: List[str] = []
    for idx, question in enumerate(questions, start=1):
        answers_serialized = " / ".join(
            f"{i + 1}:{ans}" for i, ans in enumerate(question.answers) if ans
        )
        correct_serialized = ", ".join(str(i) for i in question.correct_answers)
        explanation = question.explanation or "(설명 없음)"
        body_lines.append(
            "---\n"
            f"문항 번호: {idx}\n"
            f"질문: {question.prompt}\n"
            f"보기: {answers_serialized}\n"
            f"정답 번호: {correct_serialized}\n"
            f"설명: {explanation}\n"
        )

    body = "".join(body_lines)
    footer = (
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        "  \"reviews\": [\n"
        "    {\n"
        "      \"question_index\": 1,\n"
        "      \"verdict\": \"pass|fail|uncertain\",\n"
        "      \"issues\": \"짧은 설명 (없으면 빈 문자열)\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )
    return header + body + footer


def parse_review_response(raw_json: str, num_questions: int) -> List[dict]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise ValueError(f"검수 응답 JSON 파싱에 실패했습니다: {exc}") from exc

    reviews = payload.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("검수 응답 JSON에 'reviews' 배열이 없습니다.")

    results: List[dict] = []
    for item in reviews:
        if not isinstance(item, dict):
            continue
        index = int(item.get("question_index", 0))
        verdict = str(item.get("verdict", "")).strip().lower()
        issues = str(item.get("issues", "")).strip()
        if index < 1 or index > num_questions:
            continue
        if verdict not in REVIEW_VERDICTS:
            verdict = "uncertain"
        results.append({
            "question_index": index,
            "verdict": verdict,
            "issues": issues,
        })

    if not results:
        raise ValueError("검수 결과가 비어 있습니다.")

    results.sort(key=lambda item: item["question_index"])
    return results


def review_question_set(
    questions: Sequence[QuestionItem],
    api_key: str | None = None,
    model: str = REVIEW_MODEL,
    reference_context: str | None = None,
) -> List[dict]:
    if not questions:
        raise ValueError("검수할 문항이 없습니다.")

    api_key = ensure_api_key(api_key)
    client = OpenAI(api_key=api_key)
    prompt = build_review_prompt(questions, reference_context)

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are an impartial reviewer returning machine-readable results.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1200,
    )

    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("검수 응답이 비어 있습니다.")

    return parse_review_response(raw, len(questions))


__all__ = [
    "REVIEW_MODEL",
    "REVIEW_VERDICTS",
    "review_question_set",
]
