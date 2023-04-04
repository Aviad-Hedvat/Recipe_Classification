import joblib
from typing import Any
from Preprocessing import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

class Model:

    def __init__(self, filename: str, create: bool = False):
        self.pp = Preprocessing(filename, create)
        self.dataset = self.pp.dataset
        #self.model = self.train()
        #self.save_model(self.model, 'model.params')

    # train a svm model with Tfidf on that dataset
    def train(self):
        X = self.dataset['text']
        y = self.dataset['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
        svm_pipline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LinearSVC())
        ])
        svm_pipline.fit(X_train, y_train)
        print(classification_report(y_test, svm_pipline.predict(X_test)))
        return svm_pipline

    # save the model on a file
    def save_model(self, model, filename:str) -> None:
        joblib.dump(model, filename)

    # load a model
    def load_model(self, filename: str) -> Any:
        return joblib.load(filename)


#model = Model('dataset.csv', False)
