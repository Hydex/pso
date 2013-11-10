# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for projection='3d'


def __generate_hex_color(id):
    """
        Returns a hexadecimal color according to id
    """

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


is_init = False
fig = None
ax = None


def init(dim):
    """
        Initializes graph output
        Set is_init to True
    """

    global is_init, fig, ax

    if dim == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    is_init = True


def add_particle(particle):
    """
        Add a particle to the graph
        Required is_init == True
    """

    if not is_init:
        return

    color = __generate_hex_color(particle.id)

    if len(particle.position) == 1:
        x, y = particle.position[0], particle.fitness
        plt.plot(x, y, color=color, marker='o')
    elif len(particle.position) == 2:
        x, y, z = particle.position[0], particle.position[1], particle.fitness
        ax.scatter(x, y, z, c=color, marker='o')
    else:
        return


def show():
    """
        Shows the graph.
        Required is_init == True
    """

    if not is_init:
        return

    plt.show()
