import json

from ..config import BASE_DIR

TUNING_RESULTS_DIR = BASE_DIR / "tuning_results"


def save_tuning_result(name, best_model, best_hp):
    """튜닝 실행 결과(모델+하이퍼파라미터)를 실험별로 구분되는 파일로 저장.

    이전에는 모든 튜닝 스크립트가 같은 'best_model.keras'에 저장해서 서로
    덮어썼고, 저장된 파일이 실제 배포 모델(artifacts/)에 반영되지도 않아
    튜닝 결과가 사실상 버려졌음. 여기서는 실험 이름별로 파일을 남기되,
    배포로 승격할지는 사람이 판단하도록 artifacts/에 자동 복사하지는 않는다.
    """
    TUNING_RESULTS_DIR.mkdir(exist_ok=True)

    model_path = TUNING_RESULTS_DIR / f"{name}.keras"
    hparams_path = TUNING_RESULTS_DIR / f"{name}_hparams.json"

    best_model.save(model_path)
    with open(hparams_path, 'w', encoding='utf-8') as f:
        json.dump(best_hp.values, f, ensure_ascii=False, indent=2)

    print(f"튜닝 결과 저장됨: {model_path}")
    print(f"베스트 하이퍼파라미터 저장됨: {hparams_path}")
    print("실제 배포에 쓰려면 artifacts/로 수동 복사하고 mailbox.py의 로드 경로를 맞출 것.")

    return model_path, hparams_path
