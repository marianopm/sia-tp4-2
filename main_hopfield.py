import json
from src.HopfieldModel import *
import numpy as np

def main(): 
    with open('./config_hopfield.json', 'r') as f:
        data = json.load(f)

    letters = get_patterns("data/letters-1.txt")

    plot_patterns(letters, "Patrones")

    COUNT_LETTERS = 4
    letters_to_train = []
    idxs = np.random.choice(len(letters), size=COUNT_LETTERS, replace=False)
    for idx in idxs:
        letters_to_train.append(letters[idx])
    letters_to_train = np.array(letters_to_train)

    avg_dot_product, max_value, max_dot_product = ortogonality(letters_to_train) 
    plot_patterns(letters_to_train, f"Patrones Almacenados con Ortogonalidad {avg_dot_product} (max={max_value} , count={max_dot_product})")

    hopfield = Hopfield(letters_to_train, data['epochs'])

    random_idx = np.random.randint(len(letters_to_train))
    letter_to_mutate = letters_to_train[random_idx]
    plot_patterns(letter_to_mutate.reshape((1, len(letter_to_mutate))), f"Patron a mutar, con probabilidad {data['mutate_rate']}")

    letter_mutated = mutate(letter_to_mutate, data['mutate_rate'])
    arr_patterns, arr_energy = hopfield.predict(letter_mutated)
    plot_patterns(letter_mutated.reshape((1, len(letter_mutated))), f"Patron a mutado.")

    print()
    print(arr_patterns)
    print(arr_energy)
    plot_patterns(arr_patterns, "Prediccion de Hopfield")
    plot_energy(arr_energy)

if __name__ == "__main__":
    main()