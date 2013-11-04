#!/bin/env python2
# -*- coding: utf-8 -*-

from random import random, uniform

import numpy as np


NB_RUN = 10

NB_DIMENSIONS = 1
NB_PARTICLES = 10
ITER_MAX = 1000

V_MIN = -20.0
V_MAX = 20.0
P_MIN = -10.0
P_MAX = 10.0

W = 1
C1 = C2 = 2


def circle_problem(x):
    """
        Function reprensenting the circle problem
        x is a tuple of values where len(x) determines number of dimensions
    """

    res = 0

    for dim in xrange(len(x)):
        res += (x[dim]) ** 2

    return res


EVAL_FUNCTION = circle_problem


def limit_array(array, bounds):

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

        self.best_fitness = -1
        self.best_position = None

    def __init_position(self):

        self.position = np.array(
            [uniform(P_MIN, P_MAX) for i in xrange(NB_DIMENSIONS)]
        )

    def __init_velocity(self):

        self.velocity = np.array(
            [uniform(V_MIN, V_MAX) for i in xrange(NB_DIMENSIONS)]
        )

    def evaluate(self, func):

        fitness = func(self.position)
        if fitness < self.best_fitness or self.best_fitness == -1:
            self.best_fitness = fitness
            self.best_position = self.position

        return fitness

    def update_velocity(self, swarm_best_pos):

        r1 = random()
        r2 = random()

        inertia = self.velocity
        particle_memory = C1 * r1 * (self.best_position - self.position)
        influence = C2 * r2 * (swarm_best_pos - self.position)

        self.velocity = limit_array(
            inertia + particle_memory + influence,
            (V_MIN, V_MAX)
        )

    def update_position(self):

        self.position = limit_array(
            self.position + self.velocity,
            (P_MIN, P_MAX)
        )

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


def swarm_simulation(func):
    """
        Minimizes func by simulating the behaviour of a particle swarm
    """

    swarm = generate_swarm()
    swarm_best_pos = None
    swarm_best_fitness = -1

    for i in xrange(ITER_MAX):
        if swarm_best_fitness <= 0.001 and swarm_best_fitness != -1:
            print "Best fitness!"
            break

        for particle in swarm:
            fitness = particle.evaluate(func)
            if fitness < swarm_best_fitness or swarm_best_fitness == -1:
                swarm_best_fitness = fitness
                swarm_best_pos = particle.position.copy()

        for particle in swarm:
            particle.update_velocity(swarm_best_fitness)
            particle.update_position()
            # print particle

    return swarm_best_pos, swarm_best_fitness


if __name__ == "__main__":

    for i in xrange(NB_RUN):
        (solution, evaluation) = swarm_simulation(EVAL_FUNCTION)

        print "Solution (eval = %.2f): %s" % (evaluation, str(solution))
