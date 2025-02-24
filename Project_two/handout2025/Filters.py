

import random
import numpy as np

from models import *

import logging

class HMMFilter:
    def __init__(self, probs, tm, om, sm):
        self.__tm = tm  # Transition model
        self.__om = om  # Observation model
        self.__sm = sm  # State model
        self.__f = probs  # Initial probability distribution
        logging.basicConfig(level=logging.INFO)

    def filter(self, sensorR: int) -> np.array:
        transposed = self.__tm.get_T_transp()  # Get transposed transition matrix
        obs_probs = self.__om.get_o_reading(sensorR)  # Get observation probabilities
        self.__f = obs_probs @ transposed @ self.__f  # Update belief
        self.__f /= np.sum(self.__f)  # Normalize

        logging.info(f"Filtered probabilities: {self.__f}")
        return self.__f

class HMMSmoother:
    def __init__(self, tm, om, sm):
        self.__tm = tm  # Transition model
        self.__om = om  # Observation model
        self.__sm = sm  # State model
        logging.basicConfig(level=logging.INFO)

    def smooth(self, sensor_r_seq: np.array, f_k: np.array) -> np.array:
        beta = np.ones(self.__sm.get_num_of_states())  # Initialize beta
        T = self.__tm.get_T()  # Get transition matrix

        for i in reversed(range(len(sensor_r_seq))):
            obs_probs = self.__om.get_o_reading(sensor_r_seq[i])  # Get observation probabilities
            beta = T @ (obs_probs * beta)  # Update beta

        smoothed = f_k * beta  # Compute smoothed probabilities
        smoothed /= np.sum(smoothed)  # Normalize

        logging.info(f"Smoothed probabilities: {smoothed}")
        return smoothed