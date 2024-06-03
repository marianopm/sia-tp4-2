import math
import numpy as np
import matplotlib.pyplot as plt

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

def mutate(patterns, rate):
   mutated_letter = np.copy(patterns)
   for i in range(len(patterns)):
      if np.random.default_rng().random() < rate:
         mutated_letter[i] *= -1

   return mutated_letter

def ortogonality(patterns):
   orto_matrix = patterns.dot(patterns.T)
   np.fill_diagonal(orto_matrix, 0)

   row, _ = orto_matrix.shape
   #El producto punto promedio de los vectores.
   avg_dot_product = round(np.abs(orto_matrix).sum() / (orto_matrix.size - row), 3)
   max_value = np.abs(orto_matrix).max()
   #El numero de vectores que tienen el producto punto maximo.
   max_dot_product = np.count_nonzero(np.abs(orto_matrix) == max_value) / 2
   return avg_dot_product, max_value, max_dot_product

def plot_patterns(pattern, desc):
    num_letters = len(pattern)
    num_rows = math.ceil(math.sqrt(num_letters))
    num_cols = math.ceil(num_letters / num_rows)
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.2)
    
    #Matriz adicional para mantener la consistencia
    if num_letters == 1:
        axs = np.array([[axs]])  
    elif num_letters == 2:
        axs = axs[np.newaxis, :] if num_rows == 1 else axs[:, np.newaxis]
    
    for i, letter in enumerate(pattern):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]
        create_pattern_plot(letter, ax)

    for ax in axs.flat[num_letters:]:
        ax.remove()

    fig.suptitle(desc, fontsize=20, fontweight="bold")
    plt.show()

def create_pattern_plot(letter, ax):
    array = np.array(letter).reshape((5, 5))
    cmap = plt.cm.get_cmap('Greens')
    cmap.set_under(color='white')

    ax.imshow(array, cmap=cmap, vmin=-1, vmax=1)

    # Marco
    for i in range(6):
        ax.plot([-0.5, 4.5], [i-0.5, i-0.5], color='black', linewidth=2)
        ax.plot([i-0.5, i-0.5], [-0.5, 4.5], color='black', linewidth=2)

    for i in range(5):
        for j in range(5):
            if array[i, j] == 1:   
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=2, edgecolor='black', facecolor='none'))

    ax.axis('off')

def plot_energy(array_energy):
    plt.plot(range(len(array_energy)), array_energy, color='red')
    plt.ylabel('Energia')
    plt.xlabel("Iteraciones")
    plt.title('Función de energia')
    plt.show()