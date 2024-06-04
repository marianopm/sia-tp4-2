import numpy as np
import random
import math
from sklearn.preprocessing import StandardScaler

class Oja:
    def __init__(self, data, radius, k, learningaRate, epochs):
        self.data = data
        self.epochs = epochs
        self.radius = radius
        self.learningRate = learningaRate
        self.weights = np.random.uniform(-1, 1, data.shape[1])
        
    
    def start(self, epsilon=0.1):
        training_set = self.standarize(self.data)
        
        shuffled_list = [a for a in range(0, len(training_set))]
        random.shuffle(shuffled_list)
        
        pastWeights = self.weights
        i = 0
        while i < self.epochs and self.euclideanDistance(self.weights - pastWeights) < epsilon:
            j = 0
            while j < len(shuffled_list):
                pastWeights = self.weights
                O = np.dot(training_set[j], self.weights)
                self.weights = self.weights + self.learningRate * O * (training_set[j]-(O*self.weights))
                j += 1
            i += 1

        # normalizo los pesos
        norm_weights = self.weights
        norma2 = math.sqrt(sum(norm_weights * norm_weights))   
        norm_weights = norm_weights / norma2                   
        return norm_weights

    def euclideanDistance(self, d):
        aux = 0
        for i in range(len(d)):
            aux += (d[i])**2
        return math.sqrt(aux)
    
    def standarize(self, data):
        return StandardScaler().fit_transform(data)

