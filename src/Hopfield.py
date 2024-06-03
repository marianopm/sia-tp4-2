import numpy as np

class Hopfield:

    def __init__(self, saved_patterns, epochs) -> None:
        self.saved_patterns = saved_patterns
        self.epochs = epochs
        self.__init_weights()

    # Matriz simetrica de pesos con la diagonal 0's
    def __init_weights(self):
        _, N = self.saved_patterns.shape
        self.weights = (1 / N) * np.matmul(np.transpose(self.saved_patterns), self.saved_patterns)
        np.fill_diagonal(self.weights, 0)

    def predict(self, pattern):
        s1 = pattern
        s2 = None

        array_patterns = []
        array_energy = []
        array_patterns.append(s1)
        array_energy.append(self.calculate_energy(s1))

        iteration = 0
        stable = False
        while not stable and iteration < self.epochs:
            s2 = np.sign(np.matmul(self.weights, s1))
            self.set_zeros(s1, s2)
            array_patterns.append(s2)
            array_energy.append(self.calculate_energy(s2))

            if np.array_equal(s1, s2):
                stable = True

            s1 = s2
            iteration += 1

        return np.array(list(array_patterns)), array_energy
    
    def set_zeros(self, s1, s2):
        for indexes in np.argwhere(s2 == 0):
            s2[indexes[0]] = s1[indexes[0]]
            
    def calculate_energy(self, s1):
        return -np.dot(s1.T, np.dot(np.triu(self.weights), s1))
    
def get_patterns(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        
    pattern = []
    for i in range(26):
        letter = []
        for j in range(5):
            current_line = list(map(lambda v: int(v), lines[j + i * 5].split()))
            for n in current_line:
                letter.append(n)
        pattern.append(letter)

    pattern = np.array(pattern)
    pattern = np.where(pattern == 0, -1, pattern)

    return pattern

