import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, log_loss
from sklearn.naive_bayes import MultinomialNB

from ..data_prep import load_data, split_data


def main():
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.1, fit_prior=True)
    model.fit(X_train_tfidf, y_train)

    # NOTE: 원본 노트북 그대로 유지 - 이 루프는 alpha별 f1을 출력만 하고 최적값을
    # 따로 저장하지 않음. 루프가 끝나면 `model`은 마지막 alpha(10.0)로 학습된
    # 상태라 아래 최종 평가/저장은 "최적 alpha"가 아니라 "루프의 마지막 alpha"
    # 모델 기준. 별도 이슈로 best alpha 추적하도록 수정 예정.
    alpha_values = [0.01, 0.1, 0.5, 1.0, 10.0]
    for alpha in alpha_values:
        model = MultinomialNB(alpha=alpha, fit_prior=True)
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        f1 = f1_score(y_test, y_pred)
        print(f"Alpha: {alpha}, F1-score: {f1}")

    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)

    f1_final = f1_score(y_test, y_pred)
    loss_final = log_loss(y_test, y_pred_proba)
    print("\n 테스트 손실값: %.4f, f1 점수: %.4f" % (loss_final, f1_final))

    joblib.dump(model, 'best_model_TFNB.pkl')
    joblib.dump(tfidf_vectorizer, 'best_model_TFNB_vectorizer.pkl')

    return model, tfidf_vectorizer


if __name__ == "__main__":
    main()
