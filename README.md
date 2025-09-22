# Blooket & Gimkit Quiz Builder

Streamlit app that calls the OpenAI API to generate Korean multiple-choice questions and exports platform-ready CSV files (Blooket or Gimkit).

## Requirements

- Python 3.10+
- OpenAI API key (`OPENAI_API_KEY` environment variable or `.streamlit/secrets.toml`)

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open the app, choose the target 플랫폼(Blooket/Gimkit), enter 학년·교과·평가 내용·핵심 키워드를 입력하고 “문항 생성하기”를 누르세요. 플랫폼별 CSV는 `블루킷 CSV 다운로드` 또는 `김킷 CSV 다운로드` 버튼으로 받을 수 있습니다. 기본 템플릿은 `data/`에 포함되어 있으며 필요 시 교체할 수 있습니다.
