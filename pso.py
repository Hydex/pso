#!/bin/env python2
# -*- coding: utf-8 -*-

from random import seed, random, uniform

import numpy as np


seed(1)  # we set the random seed to 1 in order to have always the same results

NB_RUN = 10  # how many times we will run algorithm

NB_DIMENSIONS = 1  # number of dimensions of the search space
NB_PARTICLES = 100
ITER_MAX = 1000

# limits of velocity
V_MIN = -5.0
V_MAX = 5.0
# limits of the search space
P_MIN = -50.0
P_MAX = 50.0

W = 1  # weight of inertia during velocity update (not required here)
C1 = C2 = 2  # weights of particle memory and group influence


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

        self.best_fitness = -1
        self.best_position = None

    def __init_position(self):
        """
            Particle is generated in a random position
        """

        self.position = np.array(
            [uniform(P_MIN, P_MAX) for i in xrange(NB_DIMENSIONS)]
        )

    def __init_velocity(self):
        """
            Random velocity in each dimension
        """

        self.velocity = np.array(
            [uniform(V_MIN, V_MAX) for i in xrange(NB_DIMENSIONS)]
        )

    def evaluate(self, func):
        """
            Calculates fitness of func and returns the result
            The best fitness of the particle is updated if fitness is better
        """

        fitness = func(self.position)
        if fitness < self.best_fitness or self.best_fitness == -1:
            self.best_fitness = fitness
            self.best_position = self.position

        return fitness

    def update_velocity(self, swarm_best_pos):
        """
            Updates the particle velocity following this rule:
            velocity = inertia + particle memory + group influence
            Velocity is limited by V_MIN and V_MAX
        """

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
        """
            Updates particle position following this rule:
            position = previous position + velocity
            Position is limited by P_MIN and P_MAX
        """

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
        Returns the best solution and its fitness
    """

    # we generate a new swarm with random particle positions
    swarm = generate_swarm()

    swarm_best_pos = None
    swarm_best_fitness = -1

    for i in xrange(ITER_MAX):
        # if the fitness is good enough, we quit the loop
        if swarm_best_fitness <= 0.001 and swarm_best_fitness != -1:
            print "Best fitness!"
            break

        # we calculate the best fitness for a *fully connected* particle group
        for particle in swarm:
            fitness = particle.evaluate(func)
            if fitness < swarm_best_fitness or swarm_best_fitness == -1:
                swarm_best_fitness = fitness
                swarm_best_pos = particle.position

        # then we update particle positions...
        # Or should we change them in the same loop than fitness calculation?
        # It could be ok in this algo, but I prefer to be careful (wait for
        # the rest of the exercise)
        for particle in swarm:
            particle.update_velocity(swarm_best_fitness)
            particle.update_position()

    return swarm_best_pos, swarm_best_fitness


if __name__ == "__main__":

    for i in xrange(NB_RUN):
        (solution, fitness) = swarm_simulation(EVAL_FUNCTION)

        print "Solution (fitness = %.3f): %s" % (fitness, str(solution))
