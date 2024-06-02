import numpy as np
from sklearn.preprocessing import StandardScaler

class Kohonen:
    def __init__(self, data, radius, k, learningaRate):
        self.data = data
        self.dataWithoutColumns = data.iloc[:, 1:]
        self.k = k
        self.epochs = k
        self.radius = radius
        self.standarizedData = self.standarize(self.data)
        #Creo las neursonas
        self.createNeurons()
        
    def standarize(data):
        scaler = StandardScaler()
        standarized_data = scaler.fit_transform(data)
        return standarized_data
        
    def createNeurons():
        #crea las neuronas
        #inicio los pesos
        pass
        
class Neuron:
    def __init__(self):
        self.weights = 0
        self.country = None
        
    def initWeights(self):
        self.weights = np.random.uniform(low=-1, high=1, size=(1))

