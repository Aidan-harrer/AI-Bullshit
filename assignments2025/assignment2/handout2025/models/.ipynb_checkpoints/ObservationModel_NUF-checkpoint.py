# This is the observation model for the sensor with non-uniform failure probability
# 
# The observation model contains the diagonals (stored as vectors) of the observation
# matrices for each possible sensor reading r
# The last of these vectors contains the probabilities for the sensor to produce nothing ("None")
#
# The implementation follows the description as follows
# 
# the probability for
# - reporting reading r when in any of the four states (poses) i belonging to position "r" is 0.1
# - reporting r when actually in any of the four states (poses) i belonging to any of the 
#    n_Ls ∈ {3, 5, 8} existing surrounding fields Ls of r is 0.05 
# - reporting r when actually in any of the four states (poses) i belonging to any of the 
#    n_Ls2 ∈ {5, 6, 7, 9, 11, 16} existing “secondary” surrounding fields Ls2 of r is 0.025 
# - reporting "nothing" (None) is 1.0 - 0.1 - n_Ls*0.05 - n_Ls2*0.025 for ALL states i 
#
# 

import numpy as np
import matplotlib.pyplot as plt
import random

import models.StateModel

class ObservationModel:
    def __init__(self, stateModel):

        self.__stateModel = stateModel
        self.__rows, self.__cols, self.__head = stateModel.get_grid_dimensions()

        self.__dim = self.__rows * self.__cols * self.__head
        self.__num_readings = self.__rows * self.__cols + 1

        self.__vectors = np.ones(shape=(self.__num_readings, self.__dim))

        for o in range(self.__num_readings - 1):
            sx, sy = self.__stateModel.reading_to_position(o)

            for i in range(self.__dim):
                x, y = self.__stateModel.state_to_position(i)
                self.__vectors[o, i] = 0.0

                if x == sx and y == sy:
                    # "correct" reading 
                    self.__vectors[o, i] = 0.1
                elif (x == sx + 1 or x == sx - 1) and y == sy:
                    # first ring, below or above
                    self.__vectors[o, i] = 0.05
                elif (x == sx + 1 or x == sx - 1) and (y == sy + 1 or y == sy - 1):
                    # first ring, "corners"
                    self.__vectors[o, i] = 0.05
                elif x == sx and (y == sy + 1 or y == sy - 1):
                    # first ring, left or right
                    self.__vectors[o, i] = 0.05
                elif (x == sx + 2 or x == sx - 2) and (y == sy or y == sy + 1 or y == sy - 1):
                    # second ring, above / below / left / right
                    self.__vectors[o, i] = 0.025
                elif (x == sx + 2 or x == sx - 2) and (y == sy + 2 or y == sy - 2):
                    # second ring, "corners"
                    self.__vectors[o, i] = 0.025
                elif (x == sx or x == sx + 1 or x == sx - 1) and (y == sy + 2 or y == sy - 2):
                    # second ring, "horse" metric
                    self.__vectors[o, i] = 0.025

                self.__vectors[self.__num_readings - 1, i] -= self.__vectors[o, i];  # sensor reading "nothing"

    # get the number of possible sensor readings (rows * columns + 1)
    def get_nr_of_readings(self) -> int:
        return self.__num_readings

    # get the probability for the sensor to have produced reading "reading" when in state "state"
    def get_o_reading_state(self, reading: int, i: int) -> float:
        if reading == None : reading = self.__num_readings-1
        return self.__vectors[reading, i]

    # get the diagonale matrix O_reading with probabilities of the states i, i=0...nrOfStates-1 
    # to have produced reading "reading", returns a 2d-float array
    # use None for "no reading"
    def get_o_reading(self, reading: int) -> np.array(2):
        if (reading == None): reading = self.__num_readings - 1
        return np.diag( self.__vectors[reading, :])

    def sum_diags(self):
        dummy = np.diag( self.__vectors[0,:])
        for i in range(1, self.__num_readings):
            dummy += np.diag(self.__vectors[i,:])
        return dummy

    # plot the vectors as heat map(s)
    def plot_o_diags(self):
        plt.matshow(self.__vectors)
        plt.colorbar()
        plt.show()
