import fasttext
from sklearn.metrics import f1_score

from ..data_prep import load_data, split_data


def main():
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    with open("train.txt", "w") as f:
        for text, label in zip(X_train, y_train):
            f.write(f'__label__{label} {text}\n')

    model = fasttext.train_supervised(
        input="train.txt", lr=0.1, epoch=100, wordNgrams=2, dim=200, minCount=5, verbose=2
    )

    y_pred = []
    for text in X_test:
        cleaned_text = text.strip().replace("\n", "")
        predicted_label = model.predict(cleaned_text)[0][0].replace("__label__", "")
        y_pred.append(predicted_label)

    y_test_str = [str(label) for label in y_test]
    f1 = f1_score(y_test_str, y_pred, average='weighted')
    print(f"F1 Score: {f1}")

    model.save_model("best_model_FT.bin")

    return model


if __name__ == "__main__":
    main()
