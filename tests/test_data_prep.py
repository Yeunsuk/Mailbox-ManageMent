import pandas as pd

from mailbox_mgmt.data_prep import build_tokenizer, pad_train, split_data


def _sample_dataframe():
    return pd.DataFrame({
        'head': [1, 1, 0, 0, 1, 0, 1, 0],
        'body': [
            'important meeting tomorrow',
            'please review the attached document',
            'win a free prize now',
            'click here for cheap meds',
            'quarterly report attached',
            'you have won a lottery',
            'schedule confirmed for monday',
            'urgent claim your reward',
        ],
    })


def test_split_data_is_deterministic():
    df = _sample_dataframe()
    X_train1, X_test1, y_train1, y_test1 = split_data(df.copy())
    X_train2, X_test2, y_train2, y_test2 = split_data(df.copy())

    assert list(X_train1) == list(X_train2)
    assert list(X_test1) == list(X_test2)


def test_split_data_respects_test_size():
    df = _sample_dataframe()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.25)

    assert len(X_test) == 2
    assert len(X_train) == 6


def test_split_data_replaces_nan_with_placeholder():
    df = _sample_dataframe()
    df.loc[0, 'body'] = float('nan')

    X_train, X_test, y_train, y_test = split_data(df)

    assert all(isinstance(sample, str) for sample in X_train)
    assert all(isinstance(sample, str) for sample in X_test)


def test_build_tokenizer_produces_consistent_vocab_size():
    X_train = ['hello world', 'hello there', 'world of code']
    tokenizer, vocab_size, max_len, X_train_encoded = build_tokenizer(X_train)

    assert vocab_size == len(tokenizer.word_index) + 1
    assert max_len == max(len(seq) for seq in X_train_encoded)
    assert len(X_train_encoded) == len(X_train)


def test_pad_train_pads_to_requested_length():
    X_train_encoded = [[1, 2, 3], [4, 5]]
    padded = pad_train(X_train_encoded, max_len=5)

    assert padded.shape == (2, 5)
