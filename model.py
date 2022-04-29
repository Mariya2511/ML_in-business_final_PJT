import pandas as pd
import numpy as np
import dill
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

data = pd.read_csv("train.csv")
columns=['severe_toxic', 'toxic', 'obscene', 'threat', 'insult', 'identity_hate']
data['tox']=data[columns].max(axis=1).values
data = data.drop(columns = columns)
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['id','tox']),
                                                    data['tox'], random_state=0)
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]

classifier = Pipeline([('comment_text_selector', FeatureSelector(column='comment_text')),
                       ('comment_text_tfidf', TfidfVectorizer(strip_accents='unicode',
                                                              analyzer='word',
                                                              token_pattern=r'\w{1,}',
                                                              stop_words='english',
                                                              max_features=10000)),
                       ('clf', RandomForestClassifier(max_depth=2, random_state=0))])

classifier.fit(X_train, y_train)


#with open("pipeline.dill", "wb") as f:
#    dill.dump(classifier, f)