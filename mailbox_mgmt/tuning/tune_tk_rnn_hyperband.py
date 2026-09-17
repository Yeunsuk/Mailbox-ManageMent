import keras_tuner as kt
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

from ..data_prep import build_tokenizer, load_data, split_data
from ..metrics import f1_score

CHECKPOINT_PATH = 'best_model.keras'


def build_model_factory(vocab_size, max_len):
    def build_model(hp):
        model = Sequential()
        embedding_dim = hp.Int('embedding_dim', min_value=32, max_value=128, step=32)
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
        lstm_dropout = hp.Float('lstm_dropout', min_value=0.2, max_value=0.5, step=0.1)
        model.add(Dropout(lstm_dropout))

        units = hp.Int('units', min_value=64, max_value=256, step=64)
        model.add(LSTM(units, return_sequences=False, kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(lstm_dropout))

        dense_dropout = hp.Float('dense_dropout', min_value=0.2, max_value=0.5, step=0.1)
        num_dense_layers = hp.Int('num_dense_layers', 1, 4)
        for i in range(num_dense_layers):
            model.add(Dense(
                units=hp.Int(f'dense_units_{i}', min_value=64, max_value=256, step=64),
                activation='relu',
                kernel_regularizer=regularizers.l2(0.01),
            ))
            model.add(Dropout(dense_dropout))

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
    tokenizer, vocab_size, max_len, X_train_encoded = build_tokenizer(X_train)
    X_train_padded = pad_sequences(X_train_encoded, maxlen=max_len)

    tuner = kt.Hyperband(
        build_model_factory(vocab_size, max_len),
        objective='val_f1_score',
        max_epochs=100,
        factor=3,
        directory='C:\\keras_tuner',
        project_name='TK_RNN_c7',
    )

    es = EarlyStopping(monitor='val_f1_score', mode='max', verbose=1, patience=15)
    mc = ModelCheckpoint(CHECKPOINT_PATH, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    tuner.search(X_train_padded, y_train, epochs=100, validation_split=0.2, callbacks=[es, mc])

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Best hyperparameters:")
    for param, value in best_hp.values.items():
        print(f"{param}: {value}")

    return best_model, best_hp


if __name__ == "__main__":
    main()
