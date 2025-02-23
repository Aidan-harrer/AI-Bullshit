
import random
import numpy as np

from models import *


#
# Add your Filtering / Smoothing approach(es) here
#
class HMMFilter:
    def __init__(self, probs, tm, om, sm):
        self.__tm = tm
        self.__om = om
        self.__sm = sm
        self.__f = probs
        
    # sensorR is the sensor reading (index!), self._f is the probability distribution resulting from the filtering    
    def filter(self, sensorR : int) -> np.array :
        #print( self.__f)
        
        #...
        return self.__f


class HMMSmoother:
    
    def __init__(self, tm, om, sm):
        self.__tm = tm
        self.__om = om
        self.__sm = sm

    # sensor_r_seq is the sequence (array) with the t-k sensor readings for smoothing, 
    # f_k is the filtered result (f_vector) for step k
    # fb is the smoothed result (fb_vector)
    def smooth(self, sensor_r_seq : np.array, f_k : np.array) -> np.array:
        fb = self.__f # setting a dummy value here...
        # somehow compute fb to be better than f ;-)
        # ...
        return fb