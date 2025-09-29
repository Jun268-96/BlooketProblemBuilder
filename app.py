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

from rag_utils import MAX_FILE_SIZE_MB, process_uploaded_documents, retrieve_relevant_context
from review_utils import REVIEW_VERDICTS, review_question_set

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


if "rag_index" not in st.session_state:
    st.session_state["rag_index"] = {
        "vectors": None,
        "chunks": [],
        "hashes": set(),
        "sources": {},
    }
    st.session_state["rag_last_context"] = ""
if "rag_enabled" not in st.session_state:
    st.session_state["rag_enabled"] = False

if "review_results" not in st.session_state:
    st.session_state["review_results"] = []
if "review_error" not in st.session_state:
    st.session_state["review_error"] = ""


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
    rag_enabled = st.checkbox(
        "문서 기반 생성 사용",
        value=st.session_state.get("rag_enabled", False),
        help="PDF나 PPTX 학습 자료를 업로드하면 문항 생성에 참고합니다.",
    )
    st.session_state["rag_enabled"] = rag_enabled

    rag_uploads = st.file_uploader(
        "학습 자료 업로드 (PDF/PPTX)",
        type=["pdf", "pptx"],
        accept_multiple_files=True,
        help=f"각 파일은 최대 {MAX_FILE_SIZE_MB}MB까지 지원합니다.",
    )

    effective_api_key_preview = (api_key_override or "").strip() or default_api_key

    if rag_uploads:
        if not effective_api_key_preview:
            st.warning("문서 기반 생성을 사용하려면 OpenAI API 키를 입력하거나 secrets에 저장해주세요.")
        else:
            rag_index, rag_status = process_uploaded_documents(
                rag_uploads,
                effective_api_key_preview,
                st.session_state["rag_index"],
            )
            st.session_state["rag_index"] = rag_index
            for message in rag_status["added"]:
                st.success(message)
            for message in rag_status["warnings"]:
                st.warning(message)
            for message in rag_status["errors"]:
                st.error(message)

    rag_index_state = st.session_state.get("rag_index", {})
    if rag_index_state.get("sources"):
        with st.expander("처리된 학습 자료", expanded=False):
            for source in rag_index_state["sources"].values():
                summary = f"{source['name']} · {source['chunks']}개 청크"
                if source.get("truncated"):
                    summary += " (일부만 사용)"
                st.markdown(f"- {summary}")


keywords = parse_keywords(keywords_raw)

rag_index_state = st.session_state.get("rag_index", {})
if st.session_state.get("rag_enabled", False):
    if rag_index_state.get("sources"):
        st.caption("업로드한 학습 자료를 우선 참고해 문항을 생성합니다.")
    else:
        st.info("문서 기반 생성을 켰어요. 학습 자료를 업로드하면 문항에 반영됩니다.")

st.subheader("문항 생성")
st.write("생성되는 질문, 보기, 해설은 모두 한국어로 제공됩니다.")

def run_generation(
    platform: str,
    template_upload,
    api_key_override: str,
    default_api_key: str,
    grade: str,
    subject: str,
    assessment_goal: str,
    keywords: List[str],
    num_questions: int,
    model_name: str,
    creativity: float,
    time_limit: int,
) -> None:
    status_placeholder = st.empty()
    with st.spinner("문항을 생성 중입니다. 잠시만 기다려 주세요..."):
        status_placeholder.info("CSV 템플릿을 확인하는 중입니다...")
        try:
            template_columns = resolve_template_columns(platform, template_upload)
        except FileNotFoundError:
            status_placeholder.empty()
            st.error("저장된 CSV 템플릿을 찾을 수 없습니다. data/blooket_template.csv 파일을 확인해주세요.")
            return
        except Exception as exc:
            status_placeholder.empty()
            st.error(f"템플릿을 불러오는 중 오류가 발생했습니다: {exc}")
            return

        effective_api_key = (api_key_override or "").strip() or default_api_key
        st.session_state["review_results"] = []
        st.session_state["review_error"] = ""
        status_placeholder.info("OpenAI 요청을 준비하고 있어요.")

        reference_context: str | None = None
        if st.session_state.get("rag_enabled"):
            query_components = [
                grade or "",
                subject or "",
                assessment_goal or "",
                ", ".join(keywords),
                f"요청 문항 수 {num_questions}개",
            ]
            retrieval_query = " | ".join(part for part in query_components if part)
            if not retrieval_query:
                retrieval_query = "문서 기반 학습 자료"
            rag_context = retrieve_relevant_context(
                retrieval_query,
                st.session_state.get("rag_index", {}),
                effective_api_key,
                top_k=6,
            )
            if rag_context:
                reference_context = rag_context
                st.session_state["rag_last_context"] = rag_context
                status_placeholder.info("업로드한 학습 자료를 참고해 문항을 구성하고 있어요.")
            else:
                st.session_state["rag_last_context"] = ""
                status_placeholder.info("업로드한 자료에서 직접 사용할 정보를 찾지 못해 기본 정보를 활용합니다.")
        else:
            st.session_state["rag_last_context"] = ""

        try:
            status_placeholder.info("AI에 문항 생성을 요청했습니다. 잠시만 기다려 주세요...")
            questions = generate_question_set(
                grade=grade or "미정",
                subject=subject or "미정",
                assessment_goal=assessment_goal or "",
                keywords=keywords,
                num_questions=num_questions,
                model=model_name.strip() or "gpt-4o-mini",
                temperature=creativity,
                api_key=effective_api_key or None,
                reference_context=reference_context,
            )
        except Exception as exc:  # noqa: BLE001
            status_placeholder.empty()
            st.error(f"문항 생성에 실패했습니다: {exc}")
            return

        for item in questions:
            item.time_limit = time_limit

        status_placeholder.info("CSV 파일을 정리하는 중입니다...")
        try:
            csv_bytes = build_question_csv(questions, template_columns, platform)
        except Exception as exc:
            status_placeholder.empty()
            st.error(f"CSV 파일 생성에 실패했습니다: {exc}")
            return

        st.session_state["questions"] = questions
        st.session_state["export_bytes"] = csv_bytes
        st.session_state["export_platform"] = platform
        file_prefix = platform
        st.session_state["export_filename"] = (
            f"{file_prefix}_퀴즈_{grade or '학년'}_{subject or '과목'}.csv"
            .replace(" ", "_")
        )

        status_placeholder.info("생성된 문항을 자동 검수하고 있습니다...")
        try:
            review_results = review_question_set(
                questions,
                api_key=effective_api_key or None,
                model=model_name.strip() or "gpt-4o-mini",
                reference_context=reference_context,
            )
        except Exception as exc:  # noqa: BLE001
            st.session_state["review_results"] = []
            st.session_state["review_error"] = str(exc)
            status_placeholder.warning(f"문항은 생성되었지만 자동 검수에 실패했습니다: {exc}")
            st.warning("문항은 생성되었지만 자동 검수에 실패했습니다. 결과를 직접 확인해주세요.")
        else:
            st.session_state["review_results"] = review_results
            st.session_state["review_error"] = ""
            fail_count = sum(1 for item in review_results if item["verdict"] == "fail")
            uncertain_count = sum(1 for item in review_results if item["verdict"] == "uncertain")
            if fail_count:
                status_placeholder.warning(f"자동 검수에서 {fail_count}개 문항에 문제가 발견되었습니다.")
                st.warning("자동 검수에서 문제가 보고된 문항이 있습니다. 내용을 확인하고 수정한 뒤 다시 생성해주세요.")
            else:
                status_placeholder.success("문항, CSV 파일, 자동 검수까지 모두 완료되었습니다!")
                if uncertain_count:
                    st.info("일부 문항이 '불확실'로 표시되었습니다. 해당 문항을 확인해 주세요.")
                else:
                    st.success("문항 생성이 완료되었습니다. 아래에서 내용을 확인하고 파일을 내려받으세요.")


if st.button("문항 생성하기", type="primary"):
    run_generation(
        platform,
        template_upload,
        api_key_override,
        default_api_key,
        grade,
        subject,
        assessment_goal,
        keywords,
        num_questions,
        model_name,
        creativity,
        time_limit,
    )

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
    with st.expander("생성한 JSON 보기"):
        st.code(json.dumps(raw_json, ensure_ascii=False, indent=2), language="json")

    last_context = st.session_state.get("rag_last_context", "")
    if last_context:
        with st.expander("참고한 자료 요약", expanded=False):
            st.write(last_context)

    review_results_state = st.session_state.get("review_results", [])
    review_error_state = st.session_state.get("review_error", "")
    if review_error_state:
        st.warning(f"자동 검수 중 오류가 발생했습니다: {review_error_state}")

    fail_present = False
    uncertain_present = False
    if review_results_state:
        st.subheader("자동 검수 결과")
        verdict_labels = {"pass": "통과", "fail": "실패", "uncertain": "불확실"}
        review_table = []
        for item in review_results_state:
            review_table.append(
                {
                    "문항 번호": item["question_index"],
                    "결과": verdict_labels.get(item["verdict"], item["verdict"]),
                    "이슈": item.get("issues") or "-",
                }
            )
        st.dataframe(review_table, use_container_width=True)
        fail_present = any(entry["verdict"] == "fail" for entry in review_results_state)
        uncertain_present = any(entry["verdict"] == "uncertain" for entry in review_results_state)

    if fail_present:
        st.error("자동 검수에서 문제가 보고된 문항이 있습니다. 내용을 확인한 뒤 상단의 ‘문항 생성하기’ 버튼을 다시 눌러 재생성해 주세요.")
    elif uncertain_present:
        st.info("일부 문항이 '불확실'로 표시되었습니다. 해당 문항을 확인해 주세요.")

    st.caption("CSV는 현재 상태 그대로 다운로드됩니다. 수정 후에는 ‘문항 생성하기’ 버튼으로 새로 생성해 주세요.")
    if export_bytes_state and file_name_state:
        st.download_button(
            label=f"{platform_label} CSV 다운로드",
            data=export_bytes_state,
            file_name=file_name_state,
            mime="text/csv",
        )

else:
    st.info("문항을 생성하면 미리보기와 다운로드 버튼이 표시됩니다.")

