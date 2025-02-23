
import random
import numpy as np

from models import TransitionModel,ObservationModel_NUF, ObservationModel_UF,StateModel

#
# The Robot Simulator for your convenience. It keeps the current true state of the robot
# but moves and senses according to transition and observation models that are sent in when 
# they are needed accordingly
#
class RobotSim:
    def __init__(self, state, sm):
        self.__currentState = state
        #print("starting in state {}".format(state))
        self.__sm = sm
        
    def move_once( self, tm) -> int:
        newState = -1
        rand = random.random()
        
        probSum = 0.0

        for i in range(self.__sm.get_num_of_states()) :
            probSum += tm.get_T_ij( self.__currentState, i)

            if( probSum > rand) :
                newState = i
                rand = 1.0
                break

        if( newState == -1) :
            print( " no new state found ")
        
        self.__currentState = newState;
        
        return self.__currentState
    
    def sense_in_current_state(self, om) -> int:
        sensorReading = None
        
        rand = random.random()
        
        probSum = 0.0

        for i in range(self.__sm.get_num_of_readings()-1) :
            probSum += om.get_o_reading_state( i, self.__currentState)

            if( probSum > rand) :
                sensorReading = i
                rand = 1.0
                break

        return sensorReading


        
        
        
