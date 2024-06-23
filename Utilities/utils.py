"""
Provides various utility functions.

- Select a random element from a list
- Get a random integer between a min and a max
- Get all integer values present in a given string 
- Get the current date and time
- Read a file and store its lines in a list
- Normalize a value to the range 0-1 using the sigmoid function 
- Round up a number considering the number of selected decimal cases
- Round down a number considering the number of selected decimal cases
- Convert a given time value (in seconds) to the closest number of equivalent time steps (round up)
- Calculate the distance (m) that the UAV can travel per time step at maximum velocity
- Create directory for saving generated metrics
----
"""

import random
import re
import time
import math
import simParam as sp
import time
import os
import shutil
import numpy as np

def selectRandom(list):
    """
    Select a random element from a list

    :param list list: A list with a non-zero amount of elements 

    :return: Randomly selected element
    :rtype: any
    """
    # choose random element 
    selection = random.choice(list)
    return selection

def getValueInRange(min, max): 
    """ 
    Get a random integer between a minimum and a maximum

    :param int min: Minimum threshold 
    :param int max: Maximum threshold

    :return: Selected value
    :rtype: int
    """
    value = random.randint(min, max)
    return value 

def getIntFromString(string):
    """
    Get all integer values present in a string 

    :param string string: String to extract the integers from

    :return: Extracted integers
    :rtype: list[int]
    """
    # getting numbers from string
    temp = re.findall(r'\d+', string)
    # map extracted values to int 
    res = list(map(int, temp))
    # return a list with the integers 
    return res 

def currDateTime():
    """
    Get the current day and time 

    :return: Current daytime in format <Year><Month><Day><Hour><Minute>
    :rtype: string
    """
    dateTime = time.strftime("%Y%m%d_%H%M", time.localtime())
    return dateTime

def fileToList(fileName):
    """
    Read a file and store its lines in a list 

    :param string fileName: Path of the file to be read 

    :return: All lines of the given file 
    :rtype: list[str]
    """
    with open(fileName) as f:
        lines = [line.rstrip('\n') for line in f]
    return lines

def normalize(value, a, b):
    """
    Normalize a given value into the range of 0-1 using the sigmoid function.

    :param value: Value to be normalized
    :type value: float

    :return: Normalized value 
    :rtype: float
    """
    return a + (b - a) / (1 + np.exp(-value))

def roundHalfUp(n, decimals=0):
    """Round up a number considering the number of selected decimal cases

    :param n: Number to be rounded up
    :type n: float
    :param decimals: Number of decimal cases desired, defaults to 0
    :type decimals: int

    :return: The rounded up value
    :rtype: float
    """
    multiplier = 10**decimals
    return math.floor(n * multiplier + 0.5) / multiplier

def roundHalfDown(n, decimals=0):
    """Round down a number considering the number of selected decimal cases

    :param n: Number to be rounded down
    :type n: float
    :param decimals: Number of decimal cases desired, defaults to 0 
    :type decimals: int

    :return: The rounded down value 
    :rtype: float 
    """
    multiplier = 10**decimals
    return math.floor(n * multiplier) / multiplier

def timeToSteps(time):
    """Convert a given time value (in seconds) to the closest number of equivalent time steps (round up)
    
    :param time: Value to be converted
    :type time: float 

    :return: The number of time steps
    :rtype: float
    """
    # Calculate the number of time slots it will take to complete 
    result = time/(sp.T/sp.N)
    # Check if the task can be completed in a (int) amount of time steps
    re = result%1
    # Perform a round up operation of the number of time steps if necessary 
    if re >= 0.5:
        steps = roundHalfUp(result)
    else:
        steps = roundHalfUp(result) + 1
    return steps

def distancePerStep():
    """Calculate the distance (m) that the UAV can travel per time step at maximum velocity
    
    :return: Number of moves (1 m steps) the UAV can take per time step 
    :rtype: float 
    """
    # calculate the distance the uav can travel per time step 
    distance = sp.uavMaxSpeed/(sp.N/sp.T)
    # round down to the closest integer 
    distToMoves = roundHalfDown(distance, 0)
    return distToMoves

def dirSetup(savePath):
    """
    Create folder to store the generated plots based on the execution time of the script and make a copy of the used simulation parameters
    """
    time = currDateTime()
    savePath = savePath + time + '/'
    # create directory if it does not exist 
    if not os.path.isdir(savePath):
        os.mkdir(savePath)  
    # save settings used for the simulation
    shutil.copy('./Scripts/simParam.py', savePath + 'simulationParameters.txt') 
    return savePath