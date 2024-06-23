"""
Provides the various models necessary to simulate the mobile edge computing environment.

Communication Model:
    - UAV Max Speed 
    - Line of Sight Probability between a UAV and a User
    - Path Loss between a UAV and a User
    - Transmission rate of the mobile users to the UAV
    - Transmission time between a user and a UAV
    - Transmission energy between a user and a UAV
    
Offloaded computation Model:
    - Offloaded computation of a given task on a UAV
    - Offloaded computation energy for a given task on a UAV agent
    - Waiting time for a given user considering all the tasks already in the UAV's queue
    - Total offloaded computation time

Local computation model:
    - Local computation time for a given user
    - Local computation energy for a given user 

Total time and energy consumption:
    - Maximum value between the (total communication time + total offloaded computation time and the local computation time) 
    - Propulsion energy consumption for a given UAV
    - Total energy consumption between a User and a UAV

----
"""
import numpy as np 
import simParam as sp
from .mecNetwork import UAV, User, Task

######################################## Communication Model ########################################

def uavMaxSpeed(uav: UAV): 
    """
    Determine the maximum speed of the UAV 
    
    :param uav: UAV whose maximum speed is to be calculated 
    :type uav: UAV  

    :return: The maximum velocity of the UAV in m/s
    :rtype: float
    """
    # get the difference between the two positions
    difference = uav.nextPosition - uav.position 
    # calculate the norm of the difference
    l2Norm = np.linalg.norm(difference) 
    # calculate the maximum speed
    vMax = (l2Norm*sp.N)/sp.T
    return vMax

def nextMove(uav: UAV):
    """
    Determine the next movement of the UAV depending on the maximum speed 

    :param uav: UAV whose next movement is to be calculated
    :type uav: UAV 

    :return: The next movement of the UAV
    :rtype: numpy array 
    """
    # Calculate the movement the UAV performs in one time step
    movement = uav.positionDelta*(sp.T/sp.N)*sp.uavMaxSpeed
    # Calculate the next position
    nextMove = uav.position + movement
    return nextMove

def probLoS(uav: UAV, user: User):
    """
    Line of sight probability between the UAV and a mobile user

    :param class UAV: UAV 
    :param class User: User  

    :return: Line of sight probability with given inputs and simulation parameters
    :rtype: float
    """
    # calculate the first term
    eucNorm = np.linalg.norm(uav.position - user.position)
    term1 = np.arctan(sp.hUAV/(eucNorm))
    term2 = -sp.beta*(term1 - sp.alpha)
    term3 = 1 + sp.alpha**term2
    pLoS = 1/term3
    return pLoS


def pathLoss(uav: UAV, user: User):
    """
    Calculate the path-loss between a mobile user and the UAV

    :param class UAV: UAV 
    :param class User: User  

    :return: Path loss value
    :rtype: float
    """
    term1 = ((4*np.pi*sp.fc)/sp.c) * np.linalg.norm(uav.position - user.position)
    pLoS = probLoS(uav, user)
    pathLoss = 20*np.log(term1) + pLoS*sp.nLoS + (1-pLoS)*sp.nNLoS
    return pathLoss

def transmissionRate(uav: UAV, user: User):
    """
    Data transmission rate of mobile users to the UAV

    :param class UAV: UAV 
    :param class User: User  

    :return: Value for the calculated transmission rate in bits
    :rtype: float
    """
    # Convert dBm/Hz to Watt/Hz
    nZero = 10**((sp.nZero-30)/10)
    # Calculate the noise power in Watt 
    noisePower = nZero/sp.bandwidth
    term1 = sp.power*(10**(-pathLoss(uav, user)/10))
    rate = sp.bandwidth*np.log2(1 + (term1/noisePower))
    return rate

def comTransmissionTime(uav: UAV, user: User):
    """
    Transmission time for the communication model for a given UAV and user

    :param class UAV: UAV 
    :param class User: User  

    :return: Calculated transmission time in seconds
    :rtype: float
    """
    # calculate the rate of transmision 
    r = transmissionRate(uav, user)
    # get the user's task
    task = user.getTask(0)
    # calculate the transmission time
    time = task.dataSize*task.d/r
    return time

def comTransmissionEnergy(uav: UAV, user: User):
    """
    Transmission energy for the communication model for a given UAV and user

    :param class UAV: UAV 
    :param class User: User  

    :return: Value for the calculated energy consumption in watts
    :rtype: float
    """
    # calculate the rate of transmision 
    r = transmissionRate(uav, user)
    # get the user's task
    task = user.getTask(0)
    # calculate the transmission time
    if sp.writeOutput:
        print('Transmission rate: ', r)
        print('Data size: ', task.dataSize)
    energy = sp.power*(task.dataSize*task.d/r)
    return energy

######################################## Offloaded Computation Model ########################################

def offComputationTime(task: Task):
    """
    Offloaded computation time for a given task on a UAV agent

    :param class Task: A task 

    :return: Calculated offloaded computation time in seconds
    :rtype: float
    """
    time = (task.dataSize*task.d*task.cpuCycles)/sp.fUAV
    return time

def offComputationEnergy(task: Task):
    """
    Offloaded computation energy for a given task on a UAV agent

    :param class Task: A task 

    :return: Calculated offloaded energy consumption in watts
    :rtype: float
    """
    energy = sp.hwdConst*task.dataSize*task.d*task.cpuCycles*sp.fUAV
    return energy 

def waitTime(uav: UAV):
    """
    Waiting time for a given user considering all the tasks already in the UAV's queue

    :param class UAV: UAV  

    :return: Waiting time for the user in seconds
    :rtype: float
    """
    time = 0
    # get total tasks in the queue
    tasks = uav.getAllTasks()
    # iterate over tasks
    for task in tasks:
        time += offComputationTime(task)
    return time

def currentWaitTime(uav: UAV, d=0): 
    """
    Calculates the current waiting time for all tasks in the queue using the offloaded computation time 

    :param uav: UAV whose queue will be used
    :type uav: UAV 
    :param d: Task offload value, defaults to 0
    :type d: float 
    """
    time = 0
    # get total tasks in the queue
    tasks = uav.getAllTasks()
    # iterate over tasks
    for task in tasks:
        time += (task.dataSize*d*task.cpuCycles)/sp.fUAV
    return time

def offTotalTime(user: User, uav: UAV):
    """
    Total offloaded computation time

    :param class UAV: UAV 
    :param class User: User  

    :return: Calculated offloaded computation time for a given user in seconds
    :rtype: float
    """
    # get the waiting time from the tasks queued
    wTime = currentWaitTime(uav)
    # get the offloaded computation time for the current task 
    task = user.getTask(0)
    offTime = offComputationTime(task)
    offTime += wTime
    return offTime

######################################## Local Computation Model ########################################

def locComputationTime(user: User):
    """
    Local computation time for a given user

    :param class User: User  

    :return: Calculated local computation time in seconds
    :rtype: float
    """
    # get the user's task
    task: Task = user.getTask(0)
    # calculate the local computation time
    time = (task.dataSize*(1-task.d)*task.cpuCycles)/user.cpuFreq
    return time 

def locComputationEnergy(user: User):
    """
    Local computation energy for a given user 

    :param class User: User  

    :return: Calculated offloaded computation energy in watts 
    :rtype: float
    """
    # get the user's task
    task: Task = user.getTask(0)
    # calculate the local computation energy 
    energy = sp.hwdConst*task.dataSize*(1-task.d)*task.cpuCycles*user.cpuFreq**2
    return energy

######################################## Overall Time and Energy Consumption ########################################

def totalTime(uav: UAV, user: User, writeOut=False):
    """
    Maximum value between the total communication + total offloaded computation time and the local computation time 

    :param class UAV: UAV 
    :param class User: User  

    :return: Highest total time cost in seconds
    :rtype: float
    """
    # get the transmission time
    tCom = uav.transTime
    tUAV = offTotalTime(user, uav)
    # get the local computation time 
    tLoc = locComputationTime(user)
    # get the highest total time
    tTotal = max(tCom + tUAV, tLoc)
    return tTotal

def totalTimeAfterUpload(uav: UAV, user: User, uploadTime ,writeOut=False): 
    """
    Maximum value between the total offloaded computation time and the local computation time 

    :param class UAV: UAV 
    :param class User: User  

    :return: Highest total time cost in seconds
    :rtype: float
    """
    tUAV = offTotalTime(user, uav)
    # get the local computation time 
    tLoc = locComputationTime(user) - uploadTime
    # get the highest total time
    tTotal = max(tUAV, tLoc)
    # prints
    if sp.writeOutput and writeOut:
        print('---Time Values---')
        print('Offloaded time: ', tUAV)
        print('Local computation time: ', tLoc) 
        print('Total time: ', tTotal)
    return tTotal

def propulsionEnergy(uav: UAV):
    """
    Propulsion energy consumption for a given UAV

    :param class UAV: UAV 

    :return: Calculated value of the propulsion energy consumption in watts
    :rtype: float
    """
    # constant value to represent hovering energy consumption
    xi = (0.5*sp.M*sp.T)/sp.N
    # calculate the norm of the difference between positions 
    difference = uav.nextPosition - uav.position
    l2Norm = np.linalg.norm(difference) 
    term1 = (l2Norm*sp.N)/sp.T
    # calculate the propulsion energy via all the terms 
    energy = xi*(term1**2) 
    # if stationary consider hovering energy consumption
    if energy == 0:
        energy = xi
    return energy   

def totalEnergy(uav: UAV, user: User, writeOut=False):
    """
    Total energy consumption

    :param class UAV: UAV 
    :param class User: User  

    :return: Total energy cost in watts
    :rtype: float
    """
    # communication energy 
    commE = comTransmissionEnergy(uav, user)
    # offloaded computation energy
    offE = offComputationEnergy(user.getTask(0))
    # local computation energy 
    locComE = locComputationEnergy(user)
    # propulsion energy 
    propE = uav.propEnergy
    # total energy 
    tEnergy = commE + offE + locComE + propE
    # prints
    if sp.writeOutput and writeOut:
        print('---Energy Values---')
        print('Communication energy: ', commE)
        print('Offloaded computation energy: ', offE)
        print('Local comp energy: ', locComE)
        print('Propulsion energy: ', propE)
        print('Total energy: ', tEnergy)
    return tEnergy
