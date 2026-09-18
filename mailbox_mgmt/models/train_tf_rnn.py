import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GRU
from tensorflow.keras.models import Sequential

from ..data_prep import load_data, split_data
from ..metrics import f1_score


def vectorize(X_train, X_test, max_features=2000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    X_test_tfidf = vectorizer.transform(X_test).toarray()
    X_train_tfidf = X_train_tfidf.reshape(X_train_tfidf.shape[0], X_train_tfidf.shape[1], 1)
    X_test_tfidf = X_test_tfidf.reshape(X_test_tfidf.shape[0], X_test_tfidf.shape[1], 1)
    return vectorizer, X_train_tfidf, X_test_tfidf


def build_model(input_dim, num_units=32, dropout_ratio=0.4):
    model = Sequential()
    model.add(GRU(num_units, return_sequences=False, input_shape=(input_dim, 1)))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=[f1_score])
    return model


def main():
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    _, X_train_tfidf, X_test_tfidf = vectorize(X_train, X_test, max_features=2000)

    model = build_model(X_train_tfidf.shape[1])
    es = EarlyStopping(monitor='val_f1_score', mode='max', verbose=1, patience=3)
    mc = ModelCheckpoint('best_model_TFRNN.keras', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    history = model.fit(X_train_tfidf, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es, mc])

    # 학습 때 쓴 vectorizer(X_test_tfidf)로 그대로 평가 (train_tf_cnn.py와 동일한
    # 이유로 고침 - 다른 max_features의 새 vectorizer를 쓰면 안 됨).
    loss, f1 = model.evaluate(X_test_tfidf, y_test)
    print("\n 테스트 손실값: %.4f, f1점수: %.4f" % (loss, f1))

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
