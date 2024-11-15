import pandas as pd
import numpy as np
import requests
import pickle


model_url = "https://raw.github.com/Saiteja380/Machine-Learning/master/model_pickle2"

response = requests.get(model_url, stream = True)
response.raise_for_status()

model = pickle.loads(response.content)

def Predict_House_Price(area, bedrooms, age):
    input_data = [[area, bedrooms, age]]
    predicted_price = model.predict(input_data)
    return predicted_price[0]
