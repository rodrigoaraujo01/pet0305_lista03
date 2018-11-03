#!/usr/local/bin/python
# -*- coding: utf-8 -*-


import numpy as np
from scipy.stats import multivariate_normal as mvn
from minisom import MiniSom
import matplotlib.pyplot as plt


class DataGenerator(object):
    def __init__(self):
        self.full_data = np.array(
            [
                [0.88, 0.99, 0.91, 1.10],
                [0.90, 0.99, 0.93, 1.26],
                [0.90, 0.98, 0.94, 1.24],
                [0.87, 0.98, 0.97, 1.18],
                [0.93, 0.93, 0.93, 1.20],
                [0.89, 0.97, 0.92, 1.04],
                [0.88, 0.87, 0.91, 1.41],
                [0.81, 0.92, 0.80, 0.55],
                [0.82, 0.92, 0.75, 1.05],
                [0.85, 0.90, 0.64, 0.07],
                [0.77, 0.85, 0.69, -1.36],
                [0.71, 0.83, 0.72, 0.47],
                [0.75, 0.83, 0.63, -0.87],
                [0.70, 0.62, 0.60, 0.21],
                [0.44, 0.58, 0.37, -1.36],
                [0.47, 0.37, 0.45, -0.68],
                [0.23, 0.33, 0.27, -1.26],
                [0.34, 0.36, 0.51, -1.98],
                [0.31, 0.35, 0.32, -0.55],
                [0.24, 0.37, 0.36, 0.20],
                [0.76, 0.80, 0.61, 0.39],
                [0.69, 0.75, 0.68, 0.16],
                [0.24, 0.249, 0.229, 1.056],
            ]
        )
        self.countries = [
            "Reino Unido",
            "Austrália",
            "Canadá",
            "Estados Unidos",
            "Japão",
            "França",
            "Cingapura",
            "Argentina",
            "Uruguai",
            "Cuba",
            "Colômbia",
            "Brasil",
            "Paraguai",
            "Egito",
            "Nigéria",
            "Senegal",
            "Serra Leoa",
            "Angola",
            "Etiópia",
            "Moçambique",
            "China",
            "Média",
            "Desvio Padrão",
        ]


def main():
    generator = DataGenerator()
    # data normalization
    data = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, generator.full_data)

    # Initialization and training
    map_dim = 10
    som = MiniSom(map_dim, map_dim, 4, sigma=1, random_seed=1)
    som.random_weights_init(data)
    print("Training...")
    som.train_batch(data, len(data) * 1000)
    print("\n...ready!")

    plt.figure(figsize=(10, 10))
    for i, (country, dt) in enumerate(zip(generator.countries, data)):
        winnin_position = som.winner(dt)
        plt.text(
            winnin_position[0], winnin_position[1] + np.random.rand() * 0.9, country
        )
    plt.xticks(range(map_dim))
    plt.yticks(range(map_dim))
    plt.grid()
    plt.xlim([0, map_dim])
    plt.ylim([0, map_dim])
    plt.plot()
    plt.show()


if __name__ == "__main__":
    main()
