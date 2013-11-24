#!/bin/env python2
# -*- coding: utf-8 -*-

from random import seed, random, uniform
import time as t

import numpy as np


# Init list of packages
packages = []
total_value = 0
total_weight = 0
total_volume = 0
with open("pso-packages.txt") as f:
    for line in f:
        val, weight = (float(x) for x in line.strip().split(","))
        volume = uniform(1, 100)
        packages.append((val, weight, volume))

        total_value += val
        total_weight += weight
        total_volume += volume


CONTAINER_WEIGHT = 1000
CONTAINER_VOLUME = 1000

# We calculate probability we should take a package during initialization
# of our knapsack
nb_packages = len(packages)

avg_weight = total_weight / nb_packages
avg_volume = total_volume / nb_packages

avg_max_pkg = min(
    CONTAINER_WEIGHT / avg_weight,
    CONTAINER_VOLUME / avg_volume
)
PROBA_TAKE_PKG = avg_max_pkg / nb_packages


# seed(42)  # we set the random seed in order to have always the same results
# np.random.seed(42)


DECREASE_WEIGHT = False

NB_RUN = 1  # how many times we will run algorithm

NB_DIMENSIONS = nb_packages
NB_PARTICLES = 50
ITER_MAX = 1000

# limits of velocity
V_MIN, V_MAX = -4.25, 4.25

MIN_WEIGHT = 0.4
weight = 1.0  # weight of inertia during velocity update
C1 = 1.0  # weight of particle memory
C2 = 1.5  # weight of group influence


def calculate_knapsack(x):
    """
        Calculates value, weight and volume of a knapsack
        x is a boolean vector
        and x[i] == True indicates we take the packages packages[i]
    """

    num_min_dim = min(len(x), nb_packages)

    value = 0
    weight = 0
    volume = 0
    for dim in xrange(num_min_dim):
        value += packages[dim][0] * x[dim]
        weight += packages[dim][1] * x[dim]
        volume += packages[dim][2] * x[dim]

    return value, weight, volume


def knapsack_problem(x):
    """
        Returns an "evaluation" of our knapsack (x): how far we are from the
        knapsack which maximizes value and keep weight under CONTAINER_WEIGHT
    """

    value, weight, volume = calculate_knapsack(x)

    penalty = 0
    if weight > CONTAINER_WEIGHT:
        penalty += weight
    if volume > CONTAINER_VOLUME:
        penalty += volume

    return value - penalty


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
        self.best_fitness = None
        self.best_position = None

    def __init_position(self):
        """
            Particle is generated in a random position
            It is determined using probability calculated above
        """

        self.position = np.random.random(NB_DIMENSIONS) <= PROBA_TAKE_PKG

    def __init_velocity(self):
        """
            Random velocity between V_MIN and V_MAX in each dimension
        """

        self.velocity = np.empty(NB_DIMENSIONS)
        self.velocity.fill(V_MAX)

    def evaluate(self, func):
        """
            Calculates fitness of func and returns the result
            The best fitness of the particle is updated if fitness is better
        """

        self.fitness = func(self.position)
        if self.fitness > self.best_fitness or self.best_fitness is None:
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
            Updates particle position according to velocity.
            Here, velocity is the probability position[i] is True or False
            We use a sigmoid function to calculate this probability
        """

        sigmoid_vector = 1 / (1 + np.exp(self.velocity))
        self.position = np.random.random(NB_DIMENSIONS) < sigmoid_vector


def generate_swarm():
    """
        Generates NB_PARTICLES particles and returns a list of them
    """

    swarm = []

    for i in xrange(NB_PARTICLES):
        swarm.append(Particle(i + 1))

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
    swarm_best_fitness = None

    nb_iter = 0
    for nb_iter in xrange(ITER_MAX):
        # we calculate the best fitness for the particle group
        for particle in swarm:
            fitness = particle.evaluate(func)
            if fitness > swarm_best_fitness or swarm_best_fitness is None:
                swarm_best_fitness = fitness
                swarm_best_pos = particle.position

                print nb_iter, fitness

        # then we update particle positions
        for particle in swarm:
            particle.update_velocity(swarm_best_pos)
            particle.update_position()

        decrease_weight()

    return swarm_best_pos, swarm_best_fitness, nb_iter + 1


if __name__ == "__main__":

    for i in xrange(NB_RUN):
        start = t.time()

        (solution, fitness, nb_iter) = swarm_simulation(EVAL_FUNCTION)

        value, weight, volume = calculate_knapsack(solution)

        msg = "End (value = %.3f ; weight = %.3f ; " + \
              "volume = %.3f ; iteration = %d)"
        print msg % (value, weight, volume, nb_iter)

        print "Elapsed time: ", (t.time() - start), " seconds"
