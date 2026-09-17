import matplotlib.pyplot as plt

# NOTE: 원본 노트북 그대로 - 각 모델 스크립트 실행 후 콘솔에 찍힌 f1 값을 손으로
# 옮겨 적은 하드코딩된 결과. train_tf_cnn.py / train_tf_rnn.py의 평가 버그
# (다른 vectorizer로 재평가)로 인해 f1_TF_CNN, f1_TF_RNN 값은 신뢰 불가.
# 별도 이슈로 각 train_*.py가 결과를 파일/DB에 자동 기록하도록 바꿀 예정.
RESULTS = {
    'f1_TK_CNN': 0.9324,
    'f1_TF_CNN': 0.0047,
    'f1_TK_RNN': 0.9330,
    'f1_TF_RNN': 0.0009,
    'f1_TF_NB': 0.8926,
    'f1_FT': 0.9363,
    'f1_TF_LR': 0.9270,
}


def plot_results(results=RESULTS):
    models = list(results.keys())
    f1_scores = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.bar(models, f1_scores, color=['blue', 'blue', 'green', 'green', 'orange', 'red', 'purple'])
    plt.title('result')
    plt.xlabel('model')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_results()
