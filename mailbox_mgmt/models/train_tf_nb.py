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

    # alpha별로 학습해서 가장 F1이 높은 모델을 추적. 예전엔 알파별 F1을
    # 출력만 하고 최적값을 저장 안 해서, 최종 저장/평가가 "최적 alpha"가
    # 아니라 "루프의 마지막 alpha(10.0)" 기준이었음.
    alpha_values = [0.01, 0.1, 0.5, 1.0, 10.0]
    best_alpha = None
    best_f1 = -1.0
    best_model = None

    for alpha in alpha_values:
        candidate = MultinomialNB(alpha=alpha, fit_prior=True)
        candidate.fit(X_train_tfidf, y_train)
        y_pred = candidate.predict(X_test_tfidf)
        f1 = f1_score(y_test, y_pred)
        print(f"Alpha: {alpha}, F1-score: {f1}")

        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha
            best_model = candidate

    print(f"\n최적 alpha: {best_alpha} (F1: {best_f1:.4f})")

    y_pred = best_model.predict(X_test_tfidf)
    y_pred_proba = best_model.predict_proba(X_test_tfidf)
    loss_final = log_loss(y_test, y_pred_proba)
    print("\n 테스트 손실값: %.4f, f1 점수: %.4f" % (loss_final, best_f1))

    joblib.dump(best_model, 'best_model_TFNB.pkl')
    joblib.dump(tfidf_vectorizer, 'best_model_TFNB_vectorizer.pkl')

    return best_model, tfidf_vectorizer


if __name__ == "__main__":
    main()
