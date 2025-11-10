# 작업 요약

- Streamlit + Gemini 기반 문제 생성기를 구축하고, 한국어 문항을 CSV 템플릿에 채워 Blooket과 Gimkit 양식을 모두 지원하도록 확장했습니다.
- `blooket_generator.py`는 플랫폼별 템플릿 경로 관리(`data/blooket_template.csv`, `data/gimkit_template.csv`), CSV 생성 로직, JSON 파싱을 제공하며 문항/정답 매핑을 검증합니다.
- `app.py`는 플랫폼 선택 드롭다운, 템플릿 업로드(선택), JSON 미리보기, CSV 다운로드 기능을 갖춘 한국어 Streamlit UI를 구현하고, secrets/환경 변수에서 Gemini API 키를 자동 감지합니다.
- `.streamlit/secrets.toml`를 BOM 없는 UTF-8로 재저장하여 API 키 인식 문제를 해결했고, 관련 안내를 README에 반영했습니다.
- 종속성(`requirements.txt`)과 문서(`README.md`)를 최신 흐름에 맞게 업데이트했습니다.
- Blooket/Gimkit 선택 드롭다운과 김킷 CSV 템플릿을 추가해 두 플랫폼 모두에 업로드 가능한 파일을 생성합니다.
- Git 리포지토리를 초기화하고 GitHub(BlooketProblemBuilder)와 연동하여 초기 커밋을 push했습니다.
- streamlit 통해서 배포 완료했습니다.

- Streamlit UI에 "문항 생성하기" 동작 시 spinner와 단계별 안내 문구를 추가해 진행 상황을 안내하도록 개선했습니다.
- Gemini 프롬프트에 문항별 보기 수를 2~4개로 탄력적으로 구성해도 된다는 지침을 추가해 O/X 등 다양한 형태를 유도합니다.
- 프롬프트를 다시 조정해 기본은 4지선다로 생성하되, 필요할 때만 2~3지선다로 줄이도록 안내했습니다.
- PDF/PPTX 학습 자료를 업로드해 문서를 벡터화하고 RAG로 문항을 생성하도록 Streamlit UI와 백엔드를 확장했으며, 파일 크기는 8MB 이하로 제한했습니다.
- Streamlit 배포 환경에서도 바로 동작하도록 의존성에 python-pptx를 포함시키고 재배포해 모듈 누락 문제를 해결했습니다.
- 자동 검수 단계를 제거하고 사용자가 생성된 문항을 직접 확인한 뒤 바로 CSV를 내려받도록 흐름을 단순화했습니다.
