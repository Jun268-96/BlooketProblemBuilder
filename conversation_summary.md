# 작업 요약

- Streamlit + OpenAI 기반 문제 생성기를 구축하고, 한국어 문항을 CSV 템플릿에 채워 Blooket과 Gimkit 양식을 모두 지원하도록 확장했습니다.
- `blooket_generator.py`는 플랫폼별 템플릿 경로 관리(`data/blooket_template.csv`, `data/gimkit_template.csv`), CSV 생성 로직, JSON 파싱을 제공하며 문항/정답 매핑을 검증합니다.
- `app.py`는 플랫폼 선택 드롭다운, 템플릿 업로드(선택), JSON 미리보기, CSV 다운로드 기능을 갖춘 한국어 Streamlit UI를 구현하고, secrets/환경 변수에서 OpenAI API 키를 자동 감지합니다.
- `.streamlit/secrets.toml`를 BOM 없는 UTF-8로 재저장하여 API 키 인식 문제를 해결했고, 관련 안내를 README에 반영했습니다.
- 종속성(`requirements.txt`)과 문서(`README.md`)를 최신 흐름에 맞게 업데이트했습니다.
- Blooket/Gimkit 선택 드롭다운과 김킷 CSV 템플릿을 추가해 두 플랫폼 모두에 업로드 가능한 파일을 생성합니다.
