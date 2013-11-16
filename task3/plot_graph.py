# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


is_init = False


def init(dim):
    """
        Initializes graph output
        Set is_init to True
    """

    global is_init, fig, ax

    is_init = True


def add_point(x, y):
    """
        Add a point to the graph
        Required is_init == True
    """

    if not is_init:
        return

    plt.plot(x, y, color='#ff5a14', marker='o')


def show():
    """
        Shows the graph.
        Required is_init == True
    """

    if not is_init:
        return

    plt.show()
