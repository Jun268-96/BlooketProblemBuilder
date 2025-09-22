import csv
import io
import json
import os
from typing import List

import streamlit as st

from blooket_generator import (
    QuestionItem,
    build_question_csv,
    generate_question_set,
    load_template_columns,
)

PLATFORM_LABELS = {"blooket": "블루킷", "gimkit": "김킷"}
st.set_page_config(page_title="퀴즈 생성기", layout="wide")

st.title("퀴즈 생성기")
st.markdown(
    """
OpenAI API를 활용해 한국어 객관식 문제를 자동으로 만들고, 템플릿에 바로 채워 넣어 주는 도구입니다.
사이드바에 학년·교과·평가 내용·핵심 키워드를 입력한 뒤 완성된 CSV 파일을 다운로드하세요.
"""
)

# Initialize session storage
if "questions" not in st.session_state:
    st.session_state["questions"] = None
    st.session_state["export_bytes"] = None
    st.session_state["export_filename"] = None


def parse_keywords(raw: str) -> List[str]:
    cleaned = raw.replace(chr(13), "")
    normalized = cleaned.replace(chr(10), ",")
    chunks = [segment.strip() for segment in normalized.split(",")]
    return [segment for segment in chunks if segment]


def resolve_template_columns(platform: str, uploaded_file) -> List[str]:
    if uploaded_file is None:
        return load_template_columns(platform)

    data = uploaded_file.getvalue()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    header = next(reader, [])
    columns = [column.strip() for column in header if column.strip()]
    if not columns:
        raise ValueError("템플릿 CSV에서 헤더를 찾을 수 없습니다.")
    return columns


def resolve_default_api_key() -> str:
    env_key = (os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key") or "").strip()
    if env_key:
        return env_key

    secrets_obj = getattr(st, "secrets", {})  # type: ignore[attr-defined]
    direct_key = str(getattr(secrets_obj, "get", lambda *args, **kwargs: "")("openai_api_key", "")).strip()
    if direct_key:
        return direct_key

    general = getattr(secrets_obj, "get", lambda *args, **kwargs: {})("general", {})
    nested_key = str(getattr(general, "get", lambda *args, **kwargs: "")("openai_api_key", "")).strip()
    return nested_key


with st.sidebar:
    st.header("설정")
    default_api_key = resolve_default_api_key()

    platform = st.selectbox("문항을 사용할 플랫폼", options=["blooket", "gimkit"], format_func=lambda x: f"{PLATFORM_LABELS.get(x, x.title())} ({x.title()})")

    grade = st.text_input("학년", placeholder="예: 중학교 1학년")
    subject = st.text_input("교과", placeholder="예: 과학")
    assessment_goal = st.text_area(
        "평가 목표",
        placeholder="예: 물질의 상태 변화 원리 이해",
        height=80,
    )
    keywords_raw = st.text_area(
        "핵심 키워드",
        placeholder="쉼표 또는 줄바꿈으로 여러 개 입력",
        height=80,
    )
    num_questions = st.slider("문항 수", min_value=1, max_value=20, value=10)
    creativity = st.slider(
        "창의성 (temperature)", min_value=0.0, max_value=1.0, value=0.7, step=0.05
    )
    time_limit = st.number_input(
        "문항별 제한 시간 (초)", min_value=5, max_value=300, value=20, step=5
    )
    model_name = st.text_input("OpenAI 모델", value="gpt-4o-mini")
    api_key_override = st.text_input(
        "OpenAI API 키 (선택)",
        value="",
        type="password",
        help="비워 두면 secrets 또는 환경 변수 값을 사용합니다.",
    )
    template_upload = st.file_uploader(
        "CSV 템플릿 교체 (선택)",
        type=["csv"],
        help="기본 제공 템플릿 대신 사용할 CSV 헤더가 있다면 업로드하세요.",
    )

keywords = parse_keywords(keywords_raw)

st.subheader("문항 생성")
st.write("생성되는 질문, 보기, 해설은 모두 한국어로 제공됩니다.")

if st.button("문항 생성하기", type="primary"):
    try:
        template_columns = resolve_template_columns(platform, template_upload)
    except FileNotFoundError:
        st.error("내장된 CSV 템플릿을 찾을 수 없습니다. data/blooket_template.csv 파일을 확인하세요.")
    except Exception as exc:
        st.error(f"템플릿을 불러오는 중 오류가 발생했습니다: {exc}")
    else:
        effective_api_key = (api_key_override or "").strip() or default_api_key

        try:
            questions = generate_question_set(
                grade=grade or "미지정",
                subject=subject or "미지정",
                assessment_goal=assessment_goal or "",
                keywords=keywords,
                num_questions=num_questions,
                model=model_name.strip() or "gpt-4o-mini",
                temperature=creativity,
                api_key=effective_api_key or None,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"문항 생성에 실패했습니다: {exc}")
        else:
            for item in questions:
                item.time_limit = time_limit

            try:
                csv_bytes = build_question_csv(questions, template_columns, platform)
            except Exception as exc:
                st.error(f"CSV 파일 생성에 실패했습니다: {exc}")
            else:
                st.session_state["questions"] = questions
                st.session_state["export_bytes"] = csv_bytes
                st.session_state["export_platform"] = platform
                file_prefix = platform
                st.session_state["export_filename"] = (
                    f"{file_prefix}_퀴즈_{grade or '학년'}_{subject or '과목'}.csv"
                    .replace(" ", "_")
                )
                st.success("문항 생성이 완료되었습니다. 아래에서 내용을 확인하고 파일을 내려받으세요.")

questions_state: List[QuestionItem] | None = st.session_state.get("questions")
export_bytes_state: bytes | None = st.session_state.get("export_bytes")
file_name_state: str | None = st.session_state.get("export_filename")
platform_state: str = st.session_state.get("export_platform", "blooket")
platform_label = PLATFORM_LABELS.get(platform_state, platform_state.title())

if questions_state:
    records = []
    for idx, question in enumerate(questions_state, start=1):
        records.append(
            {
                "문항 번호": idx,
                "질문": question.prompt,
                "보기 1": question.answers[0] if len(question.answers) > 0 else "",
                "보기 2": question.answers[1] if len(question.answers) > 1 else "",
                "보기 3": question.answers[2] if len(question.answers) > 2 else "",
                "보기 4": question.answers[3] if len(question.answers) > 3 else "",
                "정답 번호": ", ".join(str(i) for i in question.correct_answers),
                "제한 시간(초)": question.time_limit,
                "해설": question.explanation or "",
            }
        )

    st.subheader("미리보기")
    st.dataframe(records, use_container_width=True)

    raw_json = {
        "questions": [
            {
                "prompt": q.prompt,
                "answers": q.answers,
                "correct_answers": q.correct_answers,
                "time_limit": q.time_limit,
                "explanation": q.explanation,
            }
            for q in questions_state
        ]
    }
    with st.expander("생성된 JSON 보기"):
        st.code(json.dumps(raw_json, ensure_ascii=False, indent=2), language="json")

    if export_bytes_state and file_name_state:
        st.download_button(
            label=f"{platform_label} CSV 다운로드",
            data=export_bytes_state,
            file_name=file_name_state,
            mime="text/csv",
        )
else:
    st.info("문항을 생성하면 미리보기와 다운로드 버튼이 표시됩니다.")
