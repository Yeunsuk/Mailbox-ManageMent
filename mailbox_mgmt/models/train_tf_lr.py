import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from ..data_prep import load_data, split_data


def main():
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    tfidf_vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=1,
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    model = LogisticRegression(
        C=10,
        penalty='l2',
        solver='liblinear',
        max_iter=1000,
        random_state=0,
    )
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1}")

    joblib.dump(model, 'best_model_TFLR.pkl')

    return model, tfidf_vectorizer


if __name__ == "__main__":
    main()
