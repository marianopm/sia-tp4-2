import numpy as np
import random
import math
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
        # desordeno los indices
        shuffled_list = [a for a in range(0, len(self.data))]
        random.shuffle(shuffled_list)
        
        while iterations < self.epochs:
            for i in shuffled_list:
                actualNeuronWinner, winner_row, winner_column = self.get_neuron(i)
                self.updateNeighbours(winner_row, winner_column, self.data[i], self.learningRate if iterations == 0 else 1/iterations)
            iterations +=1
        return actualNeuronWinner, winner_row, winner_column
    
    def update_neuron(self, learningRate, dataRow, neuron):
        neuron.weights += learningRate * (dataRow-neuron.weights)
        
    def updateNeighbours(self, i, j, dataRow, learningRate):
        neighbours = self.get_neighbours(i, j)
        for m in neighbours:
            self.update_neuron(learningRate, dataRow, m[0])
            
    def get_neighbours(self, i, j):
        neighbours = []
        #busco los liimites de la red
        if 0 <= i - self.radius < len(self.neuronsMatrix):
            bottomLimit = i-self.radius
        else:
            bottomLimit = 0
        if 0 <= i+self.radius < len(self.neuronsMatrix):
            upperLimit = i + self.radius
        else:
            upperLimit = len(self.neuronsMatrix)
        if 0 <= j + self.radius < len(self.neuronsMatrix[0]):
            rightLimit = j + self.radius
        else:
            rightLimit = len(self.neuronsMatrix[0])
        if 0 <= j-self.radius < len(self.neuronsMatrix[0]):
            leftLimit = j - self.radius
        else:
            leftLimit = 0

        for p in range(bottomLimit, upperLimit):
            for n in range(leftLimit, rightLimit):
                dist = math.sqrt((i-p)**2 + (j-n)**2)
                if dist <= self.radius:
                    neighbours.append((self.neuronsMatrix[p][n], p, n))

        return neighbours
    
    #busco la neurona para c/fila de dato
    def get_neuron(self, i):
        actualNeuronWinner = None
        neuron = 0
        row = 0
        winner_row = row
        winner_column = neuron
        minDistance = math.inf
        for row in range(len(self.neuronsMatrix)):
            for neuron in range(len(self.neuronsMatrix[row])):
                # calculo la distancia euclidiana.
                distance = np.linalg.norm(self.data[i] - self.neuronsMatrix[row][neuron].weights)
                if distance < minDistance:
                    minDistance = distance
                    actualNeuronWinner = self.neuronsMatrix[row][neuron]
                    winner_row = row
                    winner_column = neuron
        return actualNeuronWinner, winner_row, winner_column
    
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

