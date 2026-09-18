import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential

from ..data_prep import load_data, split_data
from ..metrics import f1_score
from ..seed import set_global_seed


def vectorize(X_train, X_test, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    X_test_tfidf = vectorizer.transform(X_test).toarray()
    X_train_tfidf = X_train_tfidf.reshape(X_train_tfidf.shape[0], X_train_tfidf.shape[1], 1)
    X_test_tfidf = X_test_tfidf.reshape(X_test_tfidf.shape[0], X_test_tfidf.shape[1], 1)
    return vectorizer, X_train_tfidf, X_test_tfidf


def build_model(input_dim, num_filters=64, kernel_size=5, dropout_ratio=0.4):
    model = Sequential()
    model.add(Conv1D(num_filters, kernel_size, padding='valid', activation='relu', input_shape=(input_dim, 1)))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(dropout_ratio))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(192, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=[f1_score])
    return model


def main():
    set_global_seed()

    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    _, X_train_tfidf, X_test_tfidf = vectorize(X_train, X_test, max_features=5000)

    model = build_model(X_train_tfidf.shape[1])
    es = EarlyStopping(monitor='val_f1_score', mode='max', verbose=1, patience=5)
    mc = ModelCheckpoint('best_model_TFCNN.keras', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    history = model.fit(X_train_tfidf, y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[es, mc])

    # 학습 때 쓴 vectorizer(X_test_tfidf)로 그대로 평가. 이전엔 max_features가 다른
    # 새 vectorizer로 다시 fit해서 모델이 학습 당시와 다른 shape/의미의 입력을
    # 받았고, 그게 evaluate 점수가 거의 0으로 나온 원인이었음.
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
