from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall

# NOTE: 원본 노트북 그대로 유지. Precision()/Recall()은 stateful metric 객체라
# Keras가 매 배치/에폭마다 자동으로 reset해주지 않음 (Metric 서브클래스가 아닌
# 순수 함수로 model.compile(metrics=[f1_score])에 전달되기 때문). 즉 여기서
# 나오는 f1_score 값은 "이번 배치 성능"이 아니라 프로세스 시작 이후 누적 평균.
# 별도 이슈로 stateless 버전으로 교체 예정 - 지금은 분리 작업만 진행.
precision_metric = Precision()
recall_metric = Recall()


def f1_score(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall + K.epsilon())
