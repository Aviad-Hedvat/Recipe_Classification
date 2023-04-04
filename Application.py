import unicodedata
from typing import Dict, List
from Model import Model
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify

class Application:

    def __init__(self):
        self.model = Model('dataset.csv')
        self.nutritions = self.model.pp.read_rename('nutrition.csv')
        self.loaded_model = self.model.load_model('svm.params')

    # get all the elements of same type from the website
    def get_all_elements_by_type(self, url: str, element_type: str) -> List:
        page = requests.get(url)
        soup = BeautifulSoup(page.text, "html.parser")
        elements = []
        for element in soup.findAll(element_type):
            prev = element.previous_sibling
            if prev and prev.name == element_type:
                element.insert(0, prev.getText() +'\n')
                prev.unwrap()

            elements.append(element.getText())

        return elements

    # perform the desired prediction
    def predict_on_elements(self, elements: List) -> np.ndarray:
        df = pd.DataFrame({'text' : elements})
        df = self.model.pp.text_preprocessing(df)
        return self.loaded_model.predict(df['text'])

    # extract the ingredients and instructions from the predictions
    def ing_and_ins(self, predictions: np.ndarray, elements: List) -> (str, str):
        ingredients = np.where(predictions == 1)
        instructions = np.where(predictions == 2)
        ing_idx = ingredients[0].tolist()[-1]
        ins_idx = instructions[0].tolist()[-1]
        return elements[ing_idx].splitlines(), elements[ins_idx]

    # create the nutritional values dictionary
    def values_by_ingredients(self, ingredients: str) -> Dict:
        values = {}

        for col in self.nutritions.columns:
            values[col] = 0

        values.pop('id')
        values.pop('name')
        values.pop('serving_size')
        for ing in ingredients:
            clean_ing = self.model.pp.manipualte_text([ing])[0]
            for i, word in enumerate(clean_ing.split(' ')):
                if i == 0:
                    quantity = 0
                    for num in word:
                        quantity += unicodedata.numeric(num)
                for name in self.nutritions['name']:
                    if word in name.lower():
                        for col in self.nutritions.columns:
                            if col in values.keys():
                                if self.nutritions[col][i] is not np.nan:
                                    val = self.nutritions[col][i]
                                    if isinstance(val, str):
                                        s = val.split(' ')
                                        val = float(s[0])
                                    values[col] += (quantity * val)
                        break

        return values

    # handling the post request of the API
    def handle_request(self, json: Dict) -> (List, Dict, str):
        elements = self.get_all_elements_by_type(json['Url'], "li")
        predictions = self.predict_on_elements(elements)
        ing, ins = self.ing_and_ins(predictions, elements)
        values = self.values_by_ingredients(ing)
        return ing, values, '\n\n'.join(ins.splitlines())



app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/AiDock', methods=['POST'])
def post_ingredients_and_instructions():
    request_data = request.get_json()
    ing, values, ins = application.handle_request(request_data)
    res_data = [{ "Recipe" : ing }, { "Nutritional information" : values }, { "INSTRUCTIONS" : ins }]
    return jsonify(res_data), 200


if __name__ == "__main__" :
    application = Application()
    app.run()
