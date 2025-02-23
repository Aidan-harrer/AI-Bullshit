# This is the observation model for a sensor with uniform failure probability
# 
# The observation model contains the diagonals (stored as vectors) of the observation
# matrices for each possible sensor reading r
# The last of these vectors contains the probabilities for the sensor to produce nothing ("None")
#
# The implementation follows the description directly, i.e.
# 
# the probability for
# - reporting reading r when in any of the four states (poses) i belonging to position "r" is 0.1
# - reporting r when actually in any of the four states (poses) i belonging to any of the 
#    n_Ls ∈ {3, 5, 8} existing surrounding fields Ls of r is 0.4/n_LS 
# - reporting r when actually in any of the four states (poses) i belonging to any of the 
#    n_Ls2 ∈ {5, 6, 7, 9, 11, 16} existing “secondary” surrounding fields Ls2 of r is 0.4/n_Ls2 
# - reporting "nothing" (None) is 0.1 for ALL states i 
#
# 

import numpy as np
import matplotlib.pyplot as plt
import random

import models.StateModel

class ObservationModelUF:
    def __init__(self, stateModel):

        self.__stateModel = stateModel
        self.__rows, self.__cols, self.__head = stateModel.get_grid_dimensions()

        self.__dim = self.__rows * self.__cols * self.__head
        self.__num_readings = self.__rows * self.__cols + 1

        self.__vectors = np.ones(shape=(self.__num_readings, self.__dim))

        for o in range(self.__num_readings - 1):
            x, y = self.__stateModel.reading_to_position(o)

            for i in range(self.__dim):
                sx, sy = self.__stateModel.state_to_position(i)
                self.__vectors[o, i] = 0.0

                if x == sx and y == sy:
                    # "correct" reading 
                    self.__vectors[o, i] = 0.1

                elif (x == sx + 1 or x == sx - 1) and y == sy:
                    # first ring, below or above
                    if ( sx == 0 and (sy == 0 or sy == self.__cols-1)) \
                        or (sx == self.__rows-1 and (sy == 0 or sy == self.__cols-1)):
                        self.__vectors[o, i] += 0.4/3
                    elif ( sx == 0 or sx == self.__rows - 1 or sy == 0 or sy == self.__cols-1):
                        self.__vectors[o, i] += 0.4/5
                    else:
                        self.__vectors[o, i] = 0.05
                elif (x == sx + 1 or x == sx - 1) and (y == sy + 1 or y == sy - 1):
                    # first ring, "corners"
                    if ( sx == 0 and (sy == 0 or sy == self.__cols-1)) \
                        or (sx == self.__rows-1 and (sy == 0 or sy == self.__cols-1)):
                        self.__vectors[o, i] += 0.4/3
                    elif ( sx == 0 or sx == self.__rows - 1 or sy == 0 or sy == self.__cols-1):
                        self.__vectors[o, i] += 0.4/5
                    else:
                        self.__vectors[o, i] = 0.05
                elif x == sx and (y == sy + 1 or y == sy - 1):
                    # first ring, left or right
                    if ( sx == 0 and (sy == 0 or sy == self.__cols-1)) \
                        or (sx == self.__rows-1 and (sy == 0 or sy == self.__cols-1)):
                        self.__vectors[o, i] += 0.4/3
                    elif ( sx == 0 or sx == self.__rows - 1 or sy == 0 or sy == self.__cols-1):
                        self.__vectors[o, i] += 0.4/5
                    else:
                        self.__vectors[o, i] = 0.05
                elif (x == sx + 2 or x == sx - 2) and (y == sy or y == sy + 1 or y == sy - 1):
                    # second ring, above / below / left / right
                    if (sx == 0 and (sy == 0 or sy == self.__cols-1)) \
                        or (sx == self.__rows-1 and (sy == 0 or sy == self.__cols-1)):
                        self.__vectors[o, i] += 0.4/5
                    elif (sx == 0 and (sy == 1 or sy == self.__cols-2)) \
                        or (sx == self.__rows-1 and (sy == 1 or sy == self.__cols-2)) \
                        or (sy == 0 and (sx == 1 or sx == self.__rows-2)) \
                        or (sy == self.__cols-1 and (sx == 1 or sx == self.__rows-2)):
                        self.__vectors[o, i] += 0.4/6
                    elif (sx == 1 and (sy == 1 or sy == self.__cols-2)) \
                        or (sx == self.__rows-2 and (sy == 1 or sy == self.__cols-2)):
                        self.__vectors[o, i] += 0.4/7
                    elif (sx == 0 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sx == self.__rows-1 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sy == 0 and (sx >= 2 and sx <= self.__rows-3)) \
                        or (sy == self.__cols-1 and (sx >= 2 and sx <= self.__rows-3)):
                        self.__vectors[o, i] += 0.4/9
                    elif (sx == 1 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sx == self.__rows-2 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sy == 1 and (sx >= 2 and sx <= self.__rows-3)) \
                        or (sy == self.__cols-2 and (sx >= 2 and sx <= self.__rows-3)):
                        self.__vectors[o, i] += 0.4/11
                    else:
                        self.__vectors[o, i] = 0.025
                elif (x == sx + 2 or x == sx - 2) and (y == sy + 2 or y == sy - 2):
                    # second ring, "corners"
                    if (sx == 0 and (sy == 0 or sy == self.__cols-1)) \
                        or (sx == self.__rows-1 and (sy == 0 or sy == self.__cols-1)):
                        self.__vectors[o, i] += 0.4/5
                    elif (sx == 0 and (sy == 1 or sy == self.__cols-2)) \
                        or (sx == self.__rows-1 and (sy == 1 or sy == self.__cols-2)) \
                        or (sy == 0 and (sx == 1 or sx == self.__rows-2)) \
                        or (sy == self.__cols-1 and (sx == 1 or sx == self.__rows-2)):
                        self.__vectors[o, i] += 0.4/6
                    elif (sx == 1 and (sy == 1 or sy == self.__cols-2)) \
                        or (sx == self.__rows-2 and (sy == 1 or sy == self.__cols-2)):
                        self.__vectors[o, i] += 0.4/7
                    elif (sx == 0 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sx == self.__rows-1 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sy == 0 and (sx >= 2 and sx <= self.__rows-3)) \
                        or (sy == self.__cols-1 and (sx >= 2 and sx <= self.__rows-3)):
                        self.__vectors[o, i] += 0.4/9
                    elif (sx == 1 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sx == self.__rows-2 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sy == 1 and (sx >= 2 and sx <= self.__rows-3)) \
                        or (sy == self.__cols-2 and (sx >= 2 and sx <= self.__rows-3)):
                        self.__vectors[o, i] += 0.4/11
                    else:
                        self.__vectors[o, i] = 0.025
                        
                elif (x == sx or x == sx + 1 or x == sx - 1) and (y == sy + 2 or y == sy - 2):
                    # second ring, "horse" metric
                    if (sx == 0 and (sy == 0 or sy == self.__cols-1)) \
                        or (sx == self.__rows-1 and (sy == 0 or sy == self.__cols-1)):
                        self.__vectors[o, i] += 0.4/5
                    elif (sx == 0 and (sy == 1 or sy == self.__cols-2)) \
                        or (sx == self.__rows-1 and (sy == 1 or sy == self.__cols-2)) \
                        or (sy == 0 and (sx == 1 or sx == self.__rows-2)) \
                        or (sy == self.__cols-1 and (sx == 1 or sx == self.__rows-2)):
                        self.__vectors[o, i] += 0.4/6
                    elif (sx == 1 and (sy == 1 or sy == self.__cols-2)) \
                        or (sx == self.__rows-2 and (sy == 1 or sy == self.__cols-2)): # sx == self.___rows-2
                        self.__vectors[o, i] += 0.4/7
                    elif (sx == 0 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sx == self.__rows-1 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sy == 0 and (sx >= 2 and sx <= self.__rows-3)) \
                        or (sy == self.__cols-1 and (sx >= 2 and sx <= self.__rows-3)):
                        self.__vectors[o, i] += 0.4/9
                    elif (sx == 1 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sx == self.__rows-2 and (sy >= 2 and sy <= self.__cols-3)) \
                        or (sy == 1 and (sx >= 2 and sx <= self.__rows-3)) \
                        or (sy == self.__cols-2 and (sx >= 2 and sx <= self.__rows-3)):
                        self.__vectors[o, i] += 0.4/11
                    else:
                        self.__vectors[o, i] = 0.025

                self.__vectors[self.__num_readings - 1, i] = 0.1;  # sensor reading "nothing"

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
