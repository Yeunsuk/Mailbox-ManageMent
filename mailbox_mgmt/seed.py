import os
import random

import numpy as np
import tensorflow as tf

from .config import SEED


def set_global_seed(seed=SEED):
    """numpy/random/tensorflow 시드 고정.

    원본 노트북은 train_test_split의 random_state만 고정돼 있고 모델 가중치
    초기화, dropout, 배치 셔플 등은 매 실행마다 달라져서 같은 코드를 다시
    돌려도 F1이 흔들렸음. 학습/튜닝 스크립트는 시작하자마자 이 함수를
    호출해서 재현성을 확보한다.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
