from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 원본 Kaggle 스팸 데이터셋 + 크롤링 데이터가 합쳐진 학습용 CSV (컬럼: head, body)
DATA_CSV = BASE_DIR / "spam(k).csv"
# mail_crawler.crawl_subjects()가 생성하는 원본(euc-kr) 크롤링 결과
RAW_CRAWL_CSV = BASE_DIR / "spam(K).csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
TOKENIZER_JSON = ARTIFACTS_DIR / "tokenizer.json"
TKRNN_MODEL = ARTIFACTS_DIR / "best_model_TKRNN.keras"
# train_tk_rnn.py가 학습 당시 계산된 max_len을 같이 저장하는 파일.
# mailbox.py는 이 값을 읽어서 패딩하므로 재학습해도 어긋나지 않음.
MODEL_CONFIG_JSON = ARTIFACTS_DIR / "model_config.json"

# model_config.json이 없을 때만 쓰는 하위호환 fallback (현재 배포된
# best_model_TKRNN.keras 학습 당시의 실제 max_len 값과 동일).
FALLBACK_MAX_LEN = 19

# train_test_split, 모델 가중치 초기화 등에 공통으로 쓰는 시드.
# seed.set_global_seed()가 이 값으로 numpy/random/tensorflow를 고정한다.
SEED = 0
