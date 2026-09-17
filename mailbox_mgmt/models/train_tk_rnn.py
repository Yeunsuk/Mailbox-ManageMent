import json

import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

from ..config import TKRNN_MODEL, TOKENIZER_JSON
from ..data_prep import build_tokenizer, load_data, split_data
from ..metrics import f1_score


def build_model(vocab_size, max_len, embedding_dim=64, dropout_ratio=0.3, units=128):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(Dropout(dropout_ratio))
    model.add(LSTM(units, return_sequences=False, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=[f1_score])
    return model


def main():
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)
    tokenizer, vocab_size, max_len, X_train_encoded = build_tokenizer(X_train)
    X_train_padded = pad_sequences(X_train_encoded, maxlen=max_len)

    # 배포용 tokenizer 저장 (best_model_TKRNN.keras가 실제 mailbox.py에서 로드하는 모델)
    tokenizer_json = tokenizer.to_json()
    with open(TOKENIZER_JSON, 'w') as f:
        json.dump(tokenizer_json, f)

    model = build_model(vocab_size, max_len)
    es = EarlyStopping(monitor='val_f1_score', mode='max', verbose=1, patience=15)
    mc = ModelCheckpoint(str(TKRNN_MODEL), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    history = model.fit(X_train_padded, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[es, mc])

    X_test_encoded = tokenizer.texts_to_sequences(X_test)
    X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len)
    loss, f1 = model.evaluate(X_test_padded, y_test)
    print("\n 테스트 손실값: %.4f, 점수: %.4f" % (loss, f1))

    epochs = range(1, len(history.history['f1_score']) + 1)
    plt.plot(epochs, history.history['f1_score'])
    plt.plot(epochs, history.history['val_f1_score'])
    plt.title('Model F1-Score')
    plt.ylabel('F1-Score')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    return model, history


if __name__ == "__main__":
    main()
