import numpy as np
import random
from sklearn.preprocessing import StandardScaler

class Kohonen:
    def __init__(self, data, radius, k, learningaRate):
        self.data = data
        self.dataWithoutColumns = data.iloc[:, 1:]
        self.k = k
        self.epochs = k
        self.radius = radius
        self.learningRate = learningaRate
        self.function = EuclideanDistance()
        self.standarized_data = self.standarize(data)
        #Creo las neursonas
        self.neuronsMatrix = self.createNeurons(self.k)
        self.init_weights(self.neuronsMatrix, data.shape[1])
        
        
    def standarize(self, data):
        scaler = StandardScaler()
        return scaler.fit_transform(data)
        
        
    def createNeurons(self, k):
        #crea las neuronas
        neurons = []
        
        for i in range(k):
            neuron_row = []
            for j in range(k):
                neuron_row.append(Neuron())
            neuron_row = np.array(neuron_row)
            neurons.append(neuron_row)
        neurons = np.array(neurons)
        
        return neurons
    
    def init_weights(self, neuronsMatrix, dim):
        for neuron_row in neuronsMatrix:
            for neuron in neuron_row:
                neuron.initWeights(dim)
        pass
        
    def start(self):
        pass
    
    def find_neuron(self):
        actualNeuronWinner = None
        winner_row = 0
        winner_column = 0
        iterations = 0
        shuffled_list = [a for a in range(0, len(self.data))]
        random.shuffle(shuffled_list)
        
        while iterations < self.epochs:
            for i in shuffled_list:
                winner, winner_row, winner_column = self.get_neuron(i)
                self.updateNeighbours(winner_row, winner_column, self.data[i], self.learningRate if iterations == 0 else 1/iterations)
            iterations += 1
        return winner, winner_row, winner_column
        
    def updateNeighbours(self, i, j, x, learningRate):
        # busca los vecinos y los actualiza
        pass
    
class Neuron:
    def __init__(self):
        self.weights = 0
        self.country = None
        
    def initWeights(self, length):
        # creo peso random -1,1
        self.weights = np.random.uniform(low=-1, high=1, size=(length,))
        
class EuclideanDistance:
    def distance(x,y):
        pass

