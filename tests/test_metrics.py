import pytest
import tensorflow as tf

from mailbox_mgmt.metrics import f1_score


def test_f1_score_perfect_prediction():
    y_true = tf.constant([1.0, 0.0, 1.0, 1.0])
    y_pred = tf.constant([0.9, 0.1, 0.8, 0.95])

    result = f1_score(y_true, y_pred)

    assert float(result) == pytest.approx(1.0, abs=1e-4)


def test_f1_score_no_positives_predicted():
    y_true = tf.constant([1.0, 1.0])
    y_pred = tf.constant([0.1, 0.2])

    result = f1_score(y_true, y_pred)

    assert float(result) == pytest.approx(0.0, abs=1e-4)


def test_f1_score_is_stateless_across_calls():
    """예전 버그: Precision()/Recall()이 stateful이라 호출할수록 값이 누적됐음.

    같은 y_true/y_pred_good 조합을 두 번 호출했을 때 그 사이에 다른 배치를
    끼워 넣어도 결과가 똑같이 나와야 stateless라고 볼 수 있다.
    """
    y_true = tf.constant([1.0, 0.0])
    y_pred_good = tf.constant([0.9, 0.1])
    y_pred_bad = tf.constant([0.1, 0.9])

    first = float(f1_score(y_true, y_pred_good))
    float(f1_score(y_true, y_pred_bad))
    third = float(f1_score(y_true, y_pred_good))

    assert first == pytest.approx(third, abs=1e-4)
    assert first == pytest.approx(1.0, abs=1e-4)
