
#
# The Localizer binds the models together and controls the update cycle in its "update" method.
# That method is called from the visualisation (Dashboard) and calls in turn the RobotSim methods as well
# as "filter(...)" in "Filters.HMMFilter"
#

import numpy as np
import matplotlib.pyplot as plt
import random

from models import *
import Filters

class Localizer:
    def __init__(self, sm, uniformF):

        self.__sm = sm

        self.__tm = TransitionModel(self.__sm)
        if uniformF == True:
            self.__om = ObservationModel_UF.ObservationModelUF(self.__sm)
        else: 
            self.__om = ObservationModel_NUF.ObservationModel(self.__sm)

        # change in initialise in case you want to start out with something else...
        self.initialise()

    # retrieve the transition model that we are currently working with
    def get_transition_model(self) -> np.array:
        return self.__tm

    # retrieve the observation model that we are currently working with
    def get_observation_model(self) -> np.array:
        return self.__om

    # the current true pose (x, h, h) that should be kept in the local variable __trueState
    def get_current_true_pose(self) -> (int, int, int):
        x, y, h = self.__sm.state_to_pose(self.__trueState)
        return x, y, h

    # the current probability distribution over all states
    def get_current_f_vector(self) -> np.array(float):
        return self.__probs

    # the current sensor reading (as position in the grid). "Nothing" is expressed as None
    def get_current_reading(self) -> (int, int):
        ret = None
        if self.__sense != None:
            ret = self.__sm.reading_to_position(self.__sense)
        return ret;

    # get the currently most likely position, based on single most probable pose
    def most_likely_position(self) -> (int, int):
        return self.__estimate

    # (re-)initialise for a new run without change of size
    def initialise(self):
        self.__trueState = random.randint(0, self.__sm.get_num_of_states() - 1)
        self.__sense = None
        self.__probs = np.ones(self.__sm.get_num_of_states()) / (self.__sm.get_num_of_states())
        self.__estimate = self.__sm.state_to_position(np.argmax(self.__probs))

        # simple evaluation measures that go into the visualisation, add more if you like!
        
        self.__rs = RobotSim( self.__trueState, self.__sm)
        self.__HMM = Filters.HMMFilter( self.__probs, self.__tm, self.__om, self.__sm)


    #
    #  The update cycle:
    #  - robot moves one step, generates new state / pose
    #  - sensor produces one reading based on the true state / pose
    #  - filtering approach produces new probability distribution based on
    #  sensor reading, transition and sensor models
    #
    #  Reports back to the caller (viewer):
    #  Return
    #  - true if sensor reading was not "nothing", else false,
    #  - AND the three values for the (new) true pose (x, y, h),
    #  - AND the two values for the (current) sensor reading (if not "nothing")
    #  - AND the error made in this step
    #  - AND the new probability distribution
    #
    def update(self) -> (bool, int, int, int, int, int, int, int, int, np.array(1)) :
        self.__trueState = self.__rs.move_once(self.__tm)
        self.__sense = self.__rs.sense_in_current_state(self.__om)
        self.__probs = self.__HMM.filter( self.__sense)
        
        fPositions = self.__probs.copy()
        
        for state in range(0, self.__sm.get_num_of_states(), 4) :
            fPositions[state:state+4] = sum(fPositions[state:state+4])
            
        self.__estimate = self.__sm.state_to_position(np.argmax(fPositions))
        
        ret = False  # in case the sensor reading is "nothing" this is kept...
        tsX, tsY, tsH = self.__sm.state_to_pose(self.__trueState)
        srX = -1
        srY = -1
        if self.__sense != None:
            srX, srY = self.__sm.reading_to_position(self.__sense)
            ret = True
            
        eX, eY = self.__estimate
       
        error = abs(tsX-eX)+abs(tsY-eY)                
                       
        
        return ret, tsX, tsY, tsH, srX, srY, eX, eY, error, fPositions

    # update in case the true pose (trajectory) is known and fed into the Localizer from an external source
    def updateWTruePose(self, trueState) -> (bool, int, int, int, int, int, int, int, int, np.array(1)) :
        self.__trueState = trueState
        self.__sense = self.__rs.sense_in_current_state()
        self.__probs = self.__HMM.filter( self.__sense)
        
        fPositions = self.__probs.copy()
        
        for state in range(0, self.__sm.get_num_of_states(), 4) :
            fPositions[state:state+4] = sum(fPositions[state:state+4])
            
      
        self.__estimate = self.__sm.state_to_position(np.argmax(fPositions))
        
        ret = False  # in case the sensor reading is "nothing" this is kept...
        tsX, tsY, tsH = self.__sm.state_to_pose(self.__trueState)
        srX = -1
        srY = -1
        if self.__sense != None:
            srX, srY = self.__sm.reading_to_position(self.__sense)
            ret = True
            
        eX, eY = self.__estimate
       
        error = abs(tsX-eX)+abs(tsY-eY)                
                       
        
        return ret, tsX, tsY, tsH, srX, srY, eX, eY, error, fPositions
