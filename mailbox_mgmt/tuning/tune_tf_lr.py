from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from ..data_prep import load_data, split_data


def main():
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('lr', LogisticRegression(max_iter=1000, random_state=0)),
    ])

    tfidf_params = {
        'tfidf__max_features': [2000, 5000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__stop_words': [None, 'english'],
        'tfidf__min_df': [1, 5],
    }
    lr_params = {
        'lr__C': [0.1, 1, 10],
        'lr__penalty': ['l2'],
        'lr__solver': ['liblinear', 'saga'],
    }

    param_grid = {**tfidf_params, **lr_params}
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1')
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")

    y_pred = grid_search.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1}")

    return grid_search


if __name__ == "__main__":
    main()
