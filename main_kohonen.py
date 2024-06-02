import json
from src.kohonenModel import *
import pandas as pd
import numpy as np

def main(): 
    with open('./config_kohonen.json', 'r') as f:
        data = json.load(f)

    data_europe = pd.read_csv('./data/europe.csv', skiprows=1, header=None)
    data_europe_without_countries = data_europe.iloc[:, 1:]
    data_europe_with_column_names = pd.read_csv('./data/europe.csv')
    
    print(data_europe)

    network = Kohonen(data_europe_without_countries, data['radius'], data['k'], data['learning_rate'])
    network.start()
    
if __name__ == "__main__":
    main()