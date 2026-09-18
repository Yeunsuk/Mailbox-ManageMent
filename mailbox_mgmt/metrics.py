import tensorflow as tf
from tensorflow.keras import backend as K


def f1_score(y_true, y_pred):
    """배치 단위로 매번 새로 계산되는 stateless F1.

    이전 버전은 Precision()/Recall() stateful 객체를 재사용해서 프로세스
    시작 이후 전체 누적 평균을 반환했음 (에폭이 지나며 오르는 것처럼 보이는
    값이 실제 성능이 아니었음). 여기서는 이번 배치의 TP/예측양성/실제양성만
    가지고 매번 새로 계산하므로 호출 간 상태를 공유하지 않는다.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)

    true_positives = tf.reduce_sum(y_true * y_pred)
    predicted_positives = tf.reduce_sum(y_pred)
    actual_positives = tf.reduce_sum(y_true)

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (actual_positives + K.epsilon())

    return 2 * (precision * recall) / (precision + recall + K.epsilon())
