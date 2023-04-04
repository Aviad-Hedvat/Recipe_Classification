import requests
import pandas as pd
import numpy as np
import string
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from typing import List

class Preprocessing:

    def __init__(self, filename: str, create: bool = False):
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation = self.punctuations_removal('&-')
        if create: # to create a new dataset
            self.df = self.read_rename(filename)
            self.dataset = self.create_dataset()
            self.dataset = self.text_preprocessing(self.dataset)
            self.dataset.to_csv('dataset.csv', index=False)
        else:
            self.dataset = pd.read_csv(filename)


    # punctuation to remove from each string
    def punctuations_removal(self, punc: str) -> str:
        punctuation = ''
        for ch in string.punctuation:
            if ch not in punc:
                punctuation += ch

        return punctuation

    # read and rename the dataset csv file
    def read_rename(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.rename(columns={'Unnamed: 0' : 'id'}, inplace=True)
        return df

    # creating the dataset with bs4 & pandas
    def create_dataset(self) -> pd.DataFrame:
        dataset = pd.DataFrame(columns=['text', 'label'])  # labels - {0 : None, 1 : ingredients, 2 : instructions}
        nulls = [np.NaN, np.NAN, np.nan]
        for i, url in enumerate(self.df['url']):
            page = requests.get(url)
            soup = BeautifulSoup(page.text, 'html.parser')
            paragraphs = soup.find_all("p")
            dataset.loc[dataset.shape[0]] = [self.df['ingredients'][i], 1]
            dataset.loc[dataset.shape[0]] = [self.df['instructions'][i], 2]

            j = 5
            while j > 0:
                s = paragraphs[np.random.randint(low=0, high=len(paragraphs))].getText()
                while s in nulls or len(s) <= 1 or (not isinstance(s, str)) or s.isspace():
                    s = paragraphs[np.random.randint(low=0, high=len(paragraphs))].getText()

                dataset.loc[dataset.shape[0]] = [s, 0]
                j -= 1

        return dataset

    # apply some functions on each string
    def text_preprocessing(self, dataset: pd.DataFrame, textCol: str = 'text') -> pd.DataFrame:
        text = list(dataset[textCol])
        dataset[textCol] = self.manipualte_text(text)
        return dataset

    # the manipulates that applies on each string
    def manipualte_text(self, text: List) -> List[str]:
        corpus = []

        for i in range(len(text)):
            r = ''.join([ch for ch in text[i] if ch not in self.punctuation])
            r = r.lower()
            r = r.split()
            r = [self.lemmatizer.lemmatize(word) for word in r]
            r = ' '.join(r)
            corpus.append(r)

        return corpus



# pp = Preprocessing()
