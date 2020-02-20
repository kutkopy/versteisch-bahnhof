import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import requests
import json

import mlflow
import mlflow.sklearn

DATA_PATH = './data/'
TRAIN_DATA = DATA_PATH + "train.csv"
TEST_DATA = DATA_PATH + "test.csv"
TEST_DATA_LABEL = DATA_PATH + "test.GOLD.csv"


def load_data():
    train_data = pd.read_csv(TRAIN_DATA)
    test_data = pd.read_csv(TEST_DATA)
    test_data_label = pd.read_csv(TEST_DATA_LABEL)

    # TODO Is list needed here?
    train_phrases = train_data['Text'].values.tolist()
    test_phrases = test_data['Text'].values.tolist()

    train_labels = train_data['Label'].values.tolist()
    test_labels = test_data_label['Prediction'].values.tolist()

    return train_phrases, train_labels, test_phrases, test_labels


def get_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    return accuracy, precision, recall, f1


def train_dialect():
    train_data, train_labels, test_data, test_labels = load_data()

    min_df = 0.0
    max_df = 1.0
    ngram_range = (1, 2)
    max_features = 2000
    n_estimators = 100

    with mlflow.start_run():
        tfidf_vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range, max_features=max_features)
        random_forest_classifier = RandomForestClassifier(n_estimators=n_estimators)
        pipeline = Pipeline([('tfidf', tfidf_vectorizer), ('clf', random_forest_classifier)])

        pipeline.fit(train_data, train_labels)
        predictions = pipeline.predict(test_data)

        accuracy, precision, recall, f1 = get_metrics(test_labels, predictions)

        mlflow.log_param("min_df", min_df)
        mlflow.log_param("max_df", max_df)
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("n_estimators", n_estimators)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(pipeline, "model")


def train_dialect_hyperparameter():

    with mlflow.start_run():

        train_data, train_labels, test_data, test_labels = load_data()

        tfidf_vectorizer = TfidfVectorizer()
        random_forest_classifier = RandomForestClassifier()
        pipeline = Pipeline([('tfidf', tfidf_vectorizer), ('clf', random_forest_classifier)])

        param_grid = {
            'tfidf__min_df': [0.0],
            'tfidf__max_df': [1.0, 2.0],
            'tfidf__ngram_range': [(1, 2), (1, 4), (1, 6)],
            'tfidf__max_features': [2000],
            'clf__n_estimators': [100]
        }

        search = GridSearchCV(pipeline, param_grid, scoring="precision_macro", cv=3)
        search.fit(train_data, train_labels)
        predictions = search.predict(test_data)

        accuracy, precision, recall, f1 = get_metrics(test_labels, predictions)

        best_pipeline = search.best_estimator_
        best_params = search.best_params_

        mlflow.log_param("min_df", best_params['tfidf__min_df'])
        mlflow.log_param("max_df", best_params['tfidf__max_df'])
        mlflow.log_param("ngram_range", best_params['tfidf__ngram_range'])
        mlflow.log_param("max_features", best_params['tfidf__max_features'])
        mlflow.log_param("n_estimators", best_params['clf__n_estimators'])

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(best_pipeline, "model")


def predict_dialect():
    url = "http://127.0.0.1:1234/invocations"
    data = {
        "columns": ["ja nei du seisch o", "jo nei du seisch au"]
    }
    headers = {'Content-type': 'application/json', 'format': 'pandas-split'}
    r = requests.post(url, data=json.dumps(data), headers=headers)
    return r.content


if __name__ == '__main__':
    train_hyperparameter()
    #train_dialect()
    #print(predict_dialect()
