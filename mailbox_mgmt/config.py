from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 원본 Kaggle 스팸 데이터셋 + 크롤링 데이터가 합쳐진 학습용 CSV (컬럼: head, body)
DATA_CSV = BASE_DIR / "spam(k).csv"
# mail_crawler.crawl_subjects()가 생성하는 원본(euc-kr) 크롤링 결과
RAW_CRAWL_CSV = BASE_DIR / "spam(K).csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
TOKENIZER_JSON = ARTIFACTS_DIR / "tokenizer.json"
TKRNN_MODEL = ARTIFACTS_DIR / "best_model_TKRNN.keras"

# tokenizer.json에 학습 당시 max_len을 같이 저장하지 않아서 하드코딩된 값.
# 원본 노트북 그대로 유지 (재학습 시 어긋날 수 있는 지점 - 별도 이슈로 처리 예정).
PREDICT_MAX_LEN = 19
