#!/bin/env python2
# -*- coding: utf-8 -*-

from random import seed, random

import numpy as np

import plot_graph as pg


# Init list of packages
packages = []
total_value = 0
with open("pso-packages.txt") as f:
    for line in f:
        val, weight = (float(x) for x in line.strip().split(","))
        packages.append((val, weight))
        total_value += val


CONTAINER_WEIGHT = 1000

seed(42)  # we set the random seed in order to have always the same results

GRAPH_OUTPUT = False
DECREASE_WEIGHT = False

NB_RUN = 1  # how many times we will run algorithm

NB_DIMENSIONS = len(packages)
NB_PARTICLES = 9
ITER_MAX = 1000

# limits of velocity and of the search space
V_MIN, V_MAX = 0.0, 1.0

MIN_WEIGHT = 0.4
weight = 1.0  # weight of inertia during velocity update
C1 = 1.5  # weight of particle memory
C2 = 2  # weight of group influence


def knapsack_problem(x):

    if len(x) != len(packages):
        raise Exception()

    value = 0
    weight = 0
    for dim in xrange(len(x)):
        value += packages[dim][0] * x[dim]
        weight += packages[dim][1] * x[dim]

    if weight <= CONTAINER_WEIGHT:
        return total_value - value
    else:
        return total_value + weight


EVAL_FUNCTION = knapsack_problem


def limit_array(array, bounds):
    """
        Limits every values of array according to bounds
        bounds[0] must be the min value and bounds[1] the max one
    """

    for x in np.nditer(array, op_flags=['readwrite']):
        if x < bounds[0]:
            x[...] = bounds[0]
        elif x > bounds[1]:
            x[...] = bounds[1]

    return array


class Particle():
    """
        Reprentation of a swarm particle
    """

    def __init__(self, id):

        self.id = id

        self.__init_position()
        self.__init_velocity()

        self.fitness = -1
        self.best_fitness = -1
        self.best_position = None

    def __init_position(self):
        """
            Particle is generated in a random position
        """

        self.position = np.array(
            [True if random() < 0.5 else False for i in xrange(NB_DIMENSIONS)],
            bool
        )

    def __init_velocity(self):
        """
            Random velocity between 0 and 1 in each dimension
        """

        self.velocity = np.array(
            [random() for i in xrange(NB_DIMENSIONS)],
            float
        )

    def evaluate(self, func):
        """
            Calculates fitness of func and returns the result
            The best fitness of the particle is updated if fitness is better
        """

        self.fitness = func(self.position)
        if self.fitness < self.best_fitness or self.best_fitness == -1:
            self.best_fitness = self.fitness
            self.best_position = self.position

        return self.fitness

    def update_velocity(self, swarm_best_pos):
        """
            Updates the particle velocity following this rule:
            velocity = inertia + particle memory + group influence
            Velocity is limited by V_MIN and V_MAX
        """

        r1 = random()
        r2 = random()

        inertia = weight * self.velocity
        particle_memory = C1 * r1 * (self.best_position - self.position)
        influence = C2 * r2 * (swarm_best_pos - self.position)

        self.velocity = limit_array(
            inertia + particle_memory + influence,
            (V_MIN, V_MAX)
        )

    def update_position(self):
        """
            Updates particle position following this rule:
            position = previous position + velocity
            Position is limited by P_MIN and P_MAX
        """

        self.position = self.velocity < 0.5

    def __str__(self):

        return "Particle #%d: position %s " % (self.id, str(self.position))


def generate_swarm():
    """
        Generates NB_PARTICLES particles and returns a list of them
    """

    swarm = []

    for i in xrange(NB_PARTICLES):
        swarm.append(Particle(i))

    return swarm


def decrease_weight():
    """
        Decreases weight in order to give less importance to inertia
        weight can decrease till MIN_WEIGHT
    """

    if DECREASE_WEIGHT:
        global weight
        weight = max(MIN_WEIGHT, weight * 0.999)


def swarm_simulation(func):
    """
        Minimizes func by simulating the behaviour of a particle swarm
        Returns the best solution and its fitness
    """

    # we generate a new swarm with random particle positions
    swarm = generate_swarm()

    swarm_best_pos = None
    swarm_best_fitness = -1

    nb_iter = 0
    for nb_iter in xrange(ITER_MAX):
        # if the fitness is good enough, we quit the loop
        if swarm_best_fitness <= 0.001 and swarm_best_fitness != -1:
            break

        # we calculate the best fitness for the particle group
        for particle in swarm:
            fitness = particle.evaluate(func)
            if fitness < swarm_best_fitness or swarm_best_fitness == -1:
                swarm_best_fitness = fitness
                swarm_best_pos = particle.position
            pg.add_particle(particle)

        # then we update particle positions
        for particle in swarm:
            particle.update_velocity(swarm_best_pos)
            particle.update_position()

        decrease_weight()

    return swarm_best_pos, swarm_best_fitness, nb_iter + 1


def calculate_knapsack(x):

    if len(x) != len(packages):
        raise Exception()

    value = 0
    weight = 0
    for dim in xrange(len(x)):
        value += packages[dim][0] * x[dim]
        weight += packages[dim][1] * x[dim]

    return value, weight


if __name__ == "__main__":

    for i in xrange(NB_RUN):
        if GRAPH_OUTPUT:
            pg.init(NB_DIMENSIONS)

        (solution, fitness, nb_iter) = swarm_simulation(EVAL_FUNCTION)

        value, weight = calculate_knapsack(solution)

        print "End (value = %.3f ; weight = %.3f ; iteration = %d)" % \
            (value, weight, nb_iter)

        pg.show()
