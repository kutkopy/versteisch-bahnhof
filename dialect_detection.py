import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DATA_PATH = './data/'
TRAIN_DATA = DATA_PATH + "train.csv"
TEST_DATA = DATA_PATH + "test.csv"
TEST_DATA_LABEL = DATA_PATH + "test.GOLD.csv"


def get_tf_idf(train_phrases, test_phrases):
    tv = TfidfVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1, 2), max_features=2500)
    train_tf_idf_features = tv.fit_transform(train_phrases)
    test_tf_idf_features = tv.transform(test_phrases)

    return train_tf_idf_features, test_tf_idf_features


def load_data():
    train_data = pd.read_csv(TRAIN_DATA)
    test_data = pd.read_csv(TEST_DATA)
    test_data_label = pd.read_csv(TEST_DATA_LABEL)

    train_phrases = train_data['Text'].values
    test_phrases = test_data['Text'].values

    train_labels = train_data['Label'].values
    test_labels = test_data_label['Prediction'].values

    train_tf_idf_features, test_tf_idf_features = get_tf_idf(train_phrases, test_phrases)
    return train_tf_idf_features, train_labels, test_tf_idf_features, test_labels


def train_predict_model(classifier, train_features, train_labels, test_features):
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    return predictions


def oracle():
    train_data, train_labels, test_data, test_labels = load_data()
    rfc = RandomForestClassifier(n_jobs=-1)
    predictions = train_predict_model(rfc, train_data, train_labels, test_data)

    print(classification_report(test_labels, predictions))


if __name__ == '__main__':
    oracle()
