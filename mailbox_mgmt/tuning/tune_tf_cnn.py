import keras_tuner as kt
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential

from ..data_prep import load_data, split_data
from ..metrics import f1_score

CHECKPOINT_PATH = 'best_model.keras'


def vectorize(X_train, X_test, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    X_test_tfidf = vectorizer.transform(X_test).toarray()
    X_train_tfidf = X_train_tfidf.reshape(X_train_tfidf.shape[0], X_train_tfidf.shape[1], 1)
    X_test_tfidf = X_test_tfidf.reshape(X_test_tfidf.shape[0], X_test_tfidf.shape[1], 1)
    return vectorizer, X_train_tfidf, X_test_tfidf


def build_model_factory(input_dim):
    def build_model(hp):
        model = Sequential()
        model.add(Conv1D(
            filters=hp.Int('num_filters', min_value=32, max_value=128, step=32),
            kernel_size=hp.Int('kernel_size', min_value=3, max_value=7, step=1),
            activation='relu', input_shape=(input_dim, 1),
        ))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(hp.Float('dropout_ratio', min_value=0.2, max_value=0.5, step=0.1)))

        for i in range(hp.Int('num_dense_layers', 3, 4)):
            model.add(Dense(
                units=hp.Int(f'dense_units_{i}', min_value=64, max_value=256, step=64),
                activation='relu',
                kernel_regularizer=regularizers.l2(0.01),
            ))
            model.add(Dropout(hp.Float('dropout_ratio', min_value=0.2, max_value=0.5, step=0.1)))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(
            optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
            loss='binary_crossentropy', metrics=[f1_score],
        )
        hp.Int('batch_size', min_value=32, max_value=128, step=32)
        return model

    return build_model


def main():
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)
    _, X_train_tfidf, X_test_tfidf = vectorize(X_train, X_test, max_features=5000)

    tuner = kt.Hyperband(
        build_model_factory(X_train_tfidf.shape[1]),
        objective='val_f1_score',
        max_epochs=10,
        factor=2,
        directory='C:\\keras_tuner',
        project_name='TF_Den_c5',
    )

    es = EarlyStopping(monitor='val_f1_score', mode='max', verbose=1, patience=15)
    mc = ModelCheckpoint(CHECKPOINT_PATH, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    tuner.search(X_train_tfidf, y_train, epochs=10, validation_split=0.2, callbacks=[es, mc])

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Best hyperparameters:")
    for param, value in best_hp.values.items():
        print(f"{param}: {value}")

    return best_model, best_hp


if __name__ == "__main__":
    main()
