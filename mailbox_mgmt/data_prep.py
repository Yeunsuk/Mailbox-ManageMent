import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from .config import DATA_CSV, SEED


def load_data(path=DATA_CSV):
    data = pd.read_csv(path, encoding='utf-8')
    data.columns = ['head', 'body']
    return data


def info(data_frame):
    data_frame.info()
    print("---------------------------------- ")
    print('Null 여부 :', data_frame.isnull().values.any())
    print('중복값 :', len(data_frame) - data_frame['body'].nunique())
    data_frame.drop_duplicates(subset=['body'], inplace=True)
    print('삭제 후 샘플 수 :', len(data_frame))
    print("---------------------------------- ")
    print(data_frame.groupby('head').size().reset_index(name='count'))


def split_data(data_frame, test_size=0.2, random_state=SEED):
    X_data = data_frame['body']
    y_data = data_frame['head']
    X_data = [sample if sample == sample else '0' for sample in X_data]
    return train_test_split(
        X_data, y_data, test_size=test_size, random_state=random_state, stratify=y_data
    )


def build_tokenizer(X_train):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train_encoded = tokenizer.texts_to_sequences(X_train)
    word_to_index = tokenizer.word_index
    vocab_size = len(word_to_index) + 1
    max_len = max(len(sample) for sample in X_train_encoded)
    return tokenizer, vocab_size, max_len, X_train_encoded


def incoding_report(tokenizer, X_train_encoded, one=2):
    word_to_index = tokenizer.word_index
    print("인코딩 결과 : ", word_to_index)
    print("등장횟수 : ", tokenizer.word_counts.items())

    total_cnt = len(word_to_index)
    rare_cnt = 0
    total_freq = 0
    rare_freq = 0

    for key, value in tokenizer.word_counts.items():
        total_freq += value
        if value < one:
            rare_cnt += 1
            rare_freq += value

    print('메일의 최대 길이 : %d' % max(len(sample) for sample in X_train_encoded))
    print('메일의 평균 길이 : %f' % (sum(map(len, X_train_encoded)) / len(X_train_encoded)))
    print('등장 빈도가 %s번 이하인 단어의 수: %s' % (one - 1, rare_cnt))
    print("단어 집합(vocabulary)에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
    print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)


def pad_train(X_train_encoded, max_len):
    return pad_sequences(X_train_encoded, maxlen=max_len)
