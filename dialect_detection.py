#!/usr/bin/env python3

import sys
import requests
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


DATA_PATH = './data/'
TRAIN_DATA = DATA_PATH + 'train.csv'
TEST_DATA = DATA_PATH + 'test.csv'
TEST_DATA_LABEL = DATA_PATH + 'test.GOLD.csv'


def load_data():
    train_data = pd.read_csv(TRAIN_DATA)
    test_data = pd.read_csv(TEST_DATA)
    test_data_label = pd.read_csv(TEST_DATA_LABEL)

    train_phrases = train_data['Text'].values
    test_phrases = test_data['Text'].values

    train_labels = train_data['Label'].values
    test_labels = test_data_label['Prediction'].values

    return train_phrases, train_labels, test_phrases, test_labels


def get_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return accuracy, precision, recall, f1


def train_dialect():
    train_data, train_labels, test_data, test_labels = load_data()

    min_df = 0.0
    max_df = 1.0
    ngram_range = (1, 2)
    max_features = 2000
    n_estimators = 100

    tfidf_vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range, max_features=max_features)
    random_forest_classifier = RandomForestClassifier(n_estimators=n_estimators)
    pipeline = Pipeline([('tfidf', tfidf_vectorizer), ('clf', random_forest_classifier)])

    print('Training random forest model...')
    pipeline.fit(train_data, train_labels)
    predictions = pipeline.predict(test_data)

    print('Evaluating the model...')
    accuracy, precision, recall, f1 = get_metrics(test_labels, predictions)

    print(f'precision: {precision:.3f}, recall: {recall:.3f}, f1-score: {f1:.3f}')


def train_dialect_hyperparameter():
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

    search = GridSearchCV(pipeline, param_grid, scoring='precision_macro', cv=3)
    print('Grid search over random forest models...')
    search.fit(train_data, train_labels)
    predictions = search.predict(test_data)

    print('Evaluating the best model...')
    accuracy, precision, recall, f1 = get_metrics(test_labels, predictions)

    best_pipeline = search.best_estimator_
    best_params = search.best_params_
    print(f'best parameters: {best_params}')

    print(f'precision: {precision:.3f}, recall: {recall:.3f}, f1-score: {f1:.3f}')


def predict_dialect():
    input_data = ['ja nei du seisch o', 'jo nei du seisch au']
    print(f'Input: {input_data}')
    url = 'http://127.0.0.1:1234/invocations'
    data = {
        'columns': input_data
    }
    headers = {'Content-type': 'application/json', 'format': 'pandas-split'}
    r = requests.post(url, data=json.dumps(data), headers=headers)
    print(f'Predictions: {r.content}')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('train', 'gridsearch', 'predict'):
        print(f'Usage: {sys.argv[0]} (train | gridsearch | predict)')
        sys.exit(1)

    if sys.argv[1] == 'train':
        print(f"started model training")
        train_dialect()
    elif sys.argv[1] == 'gridsearch':
        print(f"started grid search")
        train_dialect_hyperparameter()
    elif sys.argv[1] == 'predict':
        predict_dialect()
