import json
from src.ojaModel import *
import pandas as pd
import numpy as np

def main(): 
    with open('./config_oja.json', 'r') as f:
        data = json.load(f)

    data_europe = pd.read_csv('./data/europe.csv', skiprows=1, header=None)
    data_europe_without_countries = data_europe.iloc[:, 1:]
    data_europe_with_column_names = pd.read_csv('./data/europe.csv')
    

    network = Oja(data_europe_without_countries, data['radius'], data['k'], data['learning_rate'], data['epochs'])
    network.start()
    
if __name__ == "__main__":
    main()