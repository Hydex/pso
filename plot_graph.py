# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def __generate_hex_color(id):

    colors = (
        "#2c2c2c", "#ff5a14",
        "#1f88dd", "#47CCF3",
        "#ffc514", "#ffdf7e",
        "#86abd9", "#ffa47e",
        "#32cd32", "#86e686",
        "#fbfbd0", "#d84c7e",
        "#1abc9c", "#16a085",
        "#2ecc71", "#27ae60",
        "#3498db", "#2980b9",
        "#9b59b6", "#8e44ad",
        "#34495e", "#2c3e50",
        "#f1c40f", "#f39c12",
        "#e67e22", "#d35400",
        "#e74c3c", "#c0392b",
        "#ecf0f1", "#bdc3c7",
        "#95a5a6", "#7f8c8d"
    )

    return colors[id % len(colors)]


def add_particle(particle):
    color = __generate_hex_color(particle.id)

    x, y = 0, 0
    if len(particle.position) == 1:
        x = particle.position[0]
    elif len(particle.position) == 2:
        x, y = particle.position[0], particle.position[1]
    else:
        return

    plt.plot(x, y, color=color, marker='o')


def show():

    plt.show()