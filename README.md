# Recipe_Classification

An API application that handling 1 POST request of the following Url: http://localhost:5000/AiDock.
JSON of the request must be like:
{
  "Url" : "https://www.loveandlemons.com/pumpkin-bread/"
}

Runtime analysis:

  handle_request function - 
    get_all_elements_by_type function O(elements) O(n).
    predict_on_elements function O(text) O(n).
    ing_and_ins function O(1).
    values_by_ingredients function O(ingredients*nutriotions.shape[0]*nutritions.shape[1]) O(n^3).
 
  
