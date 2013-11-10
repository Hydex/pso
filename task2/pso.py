#!/bin/env python2
# -*- coding: utf-8 -*-

from random import seed, random, uniform

import numpy as np

import plot_graph as pg


seed(42)  # we set the random seed in order to have always the same results

GRAPH_OUTPUT = True

NB_RUN = 10  # how many times we will run algorithm

NB_DIMENSIONS = 2  # number of dimensions of the search space
NB_PARTICLES = 18
NB_NEIGHBOURS = min(3, NB_PARTICLES)
ITER_MAX = 1000

# limits of velocity and of the search space
V_MAX, P_MAX = 5.0, 100.0
V_MIN, P_MIN = (-1) * V_MAX, (-1) * P_MAX

W = 1  # weight of inertia during velocity update
C1 = 1.5  # weight of particle memory
C2 = 2  # weight of group influence


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

        self.fitness = -1
        self.best_fitness = -1
        self.best_position = None

    def __init_position(self):
        """
            Particle is generated in a random position
        """

        self.position = np.array(
            [uniform(P_MIN, P_MAX) for i in xrange(NB_DIMENSIONS)],
            float
        )

    def __init_velocity(self):
        """
            Random velocity between -1 and 1 in each dimension
        """

        self.velocity = np.array(
            [uniform(-1, 1) for i in xrange(NB_DIMENSIONS)],
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

        inertia = W * self.velocity
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

    def calculate_dist(self, particle):
        """
            Returns distance of current particle from given one
        """

        return np.linalg.norm(self.position - particle.position)

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


def __update_group(group, particle, distance):
    """
        Updates group by assigning particle if dist < all(elt[0] from group)
        group[0] is the nearest particle, group[len(group) - 1] the farest one
    """

    for i in xrange(len(group)):
        p_dist, p = group[i]
        if distance < p_dist or p_dist == -1:
            # particle is nearest than current p, so we have to
            # 1. Put particle in group[i]
            group[i] = (distance, particle)

            # 2. Move current p to the next position (and don't forget to
            #    move next particles also)
            for k in xrange(i + 1, len(group)):
                (p_dist, p), group[k] = group[k], (p_dist, p)

            # 3. Leave the loop (update is finished)
            break


def get_group_best_pos(particle, swarm, group_len=NB_NEIGHBOURS):
    """
        Returns the best position of group_len nearest neighbours of particle
    """

    # calculates group_len nearest neighbours
    # tuples composed by distance from particle and corresponding particle
    group = [(-1, None) for i in xrange(group_len)]
    for p in swarm:
        dist = particle.calculate_dist(p)
        __update_group(group, p, dist)

    # gets the best particle
    best_particle = None
    best_fitness = -1
    for dist, p in group:
        if p.fitness < best_fitness or best_fitness == -1:
            best_particle = p
            best_fitness = p.fitness

    # returns the best particle position
    return best_particle.position


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
            group_best_pos = get_group_best_pos(particle, swarm)
            particle.update_velocity(group_best_pos)
            particle.update_position()

    return swarm_best_pos, swarm_best_fitness, nb_iter + 1


if __name__ == "__main__":

    for i in xrange(NB_RUN):
        if GRAPH_OUTPUT:
            pg.init(NB_DIMENSIONS)

        (solution, fitness, nb_iter) = swarm_simulation(EVAL_FUNCTION)

        print "End (fitness = %.3f ; iteration = %d)\n%s" % \
            (fitness, nb_iter, str(solution))

        pg.show()
