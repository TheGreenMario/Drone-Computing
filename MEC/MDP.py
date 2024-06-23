"""
Module to represent the State, Action and Environment of the Markov Decision Process as classes
Additionally a TaskTag class is presented to keep track of the task currently being processed by the environment.

----
"""
from .mecNetwork import User, UAV, Task, World
import simParam as sp
import MEC.systemModel as sysModel
import math
import numpy as np
import Utilities.utils as utils 
import itertools
import torch as T

class State:
    """A class used to represent an MDP state

    :param timeSlot: Currently time slot (0 to N)
    :type timeSlot: int
    :param userFreq: Frequency of the user's device 
    :type userFreq: int 
    :param uavFreq: Frequency of the UAV's CPU 
    :type uavFreq: int
    :param dataRate: Data transmission rate 
    :type dataRate: float 
    :param setOmega: Information regarding the task of a given mobile user
    :type setOmega: list(int)
    :param uavPosition: Current position of the UAV
    :type uavPosition: numpy array
    :param tasksQueue: All tasks in the UAV's queue 
    :type tasksQueue: list(Task)
    :param normalized: Array with the normalized values of the state variables
    :type normalized: numpy array 
    """

    # create state with given parameters
    def __init__(self, uav: UAV, timeSlot: int, user: User = None):
        """Constructor method to initialize the instance with the given uav and user pair and the current time slot 

        :param uav: UAV instance associated with the state
        :type uav: UAV  
        :param timeSlot: current time step 
        :type timeSlot: int 
        :param user: User associated with the task being processed, default to None 
        :type user: User  
        """
        # set state time slot
        self.timeSlot = timeSlot
        if user != None:
            # get device frequency of the given user 
            self.userFreq = user.cpuFreq
            # data transfer speed 
            self.dataRate = sysModel.transmissionRate(uav, user)
            # information of given task on given mobile user 
            self.setOmega(user.getTask(0))
            self.userPositon = np.array([user.getX(), user.getY()])
        else:
            self.userFreq = 0
            self.dataRate = 0
            self.setOmega()
            self.userPositon = np.array([0, 0])
        # device frequency of the UAV
        self.uavFreq = sp.fUAV
        # current UAV position
        self.uavPosition = uav.position
        # tasks queue states in the UAV at the given time slot 
        self.tasksQueue = uav.getAllTasks() 

    def setOmega(self, task = None):
        """Create a list to store the three omega values regarding the task's information

        :param task: The task whose information needs to be added, defaults to None
        :type task: Task 
        """
        # initialize empty list
        self.omega = []
        if task != None:
            # bits of data needed to compute for the given user
            self.omega.append(task.dataSize)
            # required cpu cycles to compute per bit data
            self.omega.append(task.cpuCycles)
            # maximum time tolerance for the task
            self.omega.append(task.tolerance)
        else:
            # bits of data needed to compute for the given user
            self.omega.append(0)
            # required cpu cycles to compute per bit data
            self.omega.append(0)
            # maximum time tolerance for the task
            self.omega.append(0)

    def toTensor(self):
        """
        Normalize the state parameters using the sigmoid function.
        """ 
        # Build the tensor 
        self.normalized = T.tensor(np.array([self.userFreq \
                                    , self.dataRate, self.uavPosition[0], self.uavPosition[1] \
                                    , self.userPositon[0], self.userPositon[1] \
                                    , self.omega[0], self.omega[1], self.omega[2], len(self.tasksQueue)]), dtype=T.float64)
        return self.normalized
        
    # helper function to print task parameters
    def __str__(self):
        """print States's time slot, user frequency, UAV frequency, data transmission rate, task data size, task data CPU cycles, task tolerance and UAV position 
        """
        return f"Time Slot: {self.timeSlot}, User Freq: {self.userFreq}, Data Rate: {self.dataRate}, DataSize: {self.omega[0]}, CpuCycles: {self.omega[1]}, TaskTolerance: {self.omega[2]}, UAV Position: {self.uavPosition}"


class Action:
    """Class to represent a UAV action 

    :param taskOffload: Task percentage to be offloaded to the UAV  
    :type taskOffload: float
    :param positionDelta: Next movement of the UAV
    :type positionDelta: numpy array
    :param actionSpace: Space of all 45 possible actions the UAV can take, where each action is a list of [d, movement]
    :type actionSpace: list[list[d, movement]]
    """

    def __init__(self, taskOffload = 0, positionDelta = None):
        """Constructor method to initialize the instance with the given task offloading percentage and position delta
        """
        # initialize the action space 
        self.createSpace()
        # if values are given
        if taskOffload != 0 and positionDelta.all():
            self.taskOffload = taskOffload
            self.positionDelta = positionDelta
        # generate a random action
        else:
            self.fromRandomIndex()

    def createSpace(self):
        """Create a list with all possible task offloading and movement combinations to form the action space
        """
        # task offloading space
        d = sp.taskOffloading
        movement = sp.movementOptions
        iterables = [ d, movement ]
        actionSpace = []
        # cartesian product of the possible movement and task offloading actions 
        for t in itertools.product(*iterables):
            actionSpace.append(t)
        # set action space 
        self.actionSpace = actionSpace

    def fromRandomIndex(self):
        """Get a random action by generating a random index and seeing what it corresponds to on the action space list 
        """
        # generate a random index from the action space 
        actionIndex = utils.getValueInRange(0, len(self.actionSpace)-1)
        # get the corresponding action
        action = self.fromIndex(actionIndex)
        self.index = actionIndex
        return action

    def fromIndex(self, index: int):
        """Select action from the action space based on the given index

        :param index: Index that represents the action chosen 
        :type index: int
        """
        # get the corresponding action
        actions = self.actionSpace[index]
        # set the action parameters 
        self.taskOffload = actions[0]
        self.positionDelta = np.array(actions[1])
        self.index = index
       
    def __str__(self):
        """print the action's task offloading percentage and position delta
        """
        return f"Task offload: {self.taskOffload}, Position Delta: {self.positionDelta}"

class Environment:
    """Class to represent the MDP environment 

    :param timeSlot: Keep track of the current time step
    :type timeSlot: int 
    :param state: List to store each State 
    :type state: list[State]
    :param world: Parameter to represent world where the user's and the uav take actions
    :type world: World 
    """

    def __init__(self):
        """Constructor method to initialize the MEC environment
        """
        # create empty list to store states 
        self.states = []
        # create new world 
        self.world = World()
    
    def incrementTime(self):
        """Increment the environment's time step by 1 
        """
        self.timeSlot += 1 
        self.tag.timeStep = self.timeSlot

    def addTag(self, uav: UAV):
        """Add tag for task tracking
        """
        self.tag = TaskTag(uav, self.world)
        
    def step(self):
        """Move to the next step in the environment by updating the time
        """
        # increment time step
        self.incrementTime()

    def startWorld(self, load=False):
        """Spawn the user's in the environment's world 

        :param load: Wether user's positions should be loaded from a file or not 
        :type load: boolean
        """
        # spawn users
        if not load:
            # select bounds according to map type 
            match sp.mapType:
                case 'square':
                    bounds = [sp.minBound,sp.maxBound,sp.minBound,sp.maxBound]
                case 'rectangle':
                    bounds = [sp.x1,sp.x2,sp.y1,sp.y2]
            self.world.spawn(sp.totalUsers, bounds)
        else:
            # load user locations from a file 
            users = utils.fileToList('./output/users.txt')
            # process user information to extract the positions
            for index in range(len(users)):
                item = utils.getIntFromString(users[index])
                # generate specified user 
                self.world.generateAgent('User', item)
           
    # check if a new task is generated for a new time slot or force the generation of one 
    def generateTask(self, uav: UAV, timeStep: int, force=False):
        """Create a task with (odds)/100 probability, or forcefully create one 

        :param uav: UAV to check the queu status before generating the task 
        :type uav: UAV
        :param timeStep: Current episode time step 
        :type: int 
        :param force: True if a task should be generated independently of odds, defaults to False
        :type force: boolean

        :return: 1: task created | 0: task not created due to odds | -1: UAV queue is full
        :rtype: int 
        """
        # guarantee the creation of a task 
        if force:
            odds = sp.taskThreshold
        # get a random number to represent the task creation probability
        else:
            odds = utils.getValueInRange(1, 100)
        # check if the generated number is higher than the task creation threshold 
        if odds <= sp.taskThreshold:
            # get a list of all users
            users = self.world.users
            # get total number of users
            totalUsers = len(users)
            # get random user 
            random = utils.getValueInRange(0, totalUsers-1)
            # check if chosen user has a task
            user = users[random]
            # create a task for that user 
            user.createTask(timeStep)
            # add task to queue
            uav.addTask(user.getTask(len(user.tasks)-1)) 
            # new task created 
            if sp.writeOutput:
                print('NEW TASK RANDOMLY GENERATED for user: ', user)
            return 1
        else:
            # new task not created
            if sp.writeOutput: 
                print('Task not created due to generation odds')
            return 0
   
    def checkExpiration(self, uav: UAV, user: User): 
        """Check if the task of a user can be completed before the expiration time, considering 
        the total processing time cost 

        :param uav: UAV whose queue should be used 
        :type uav: UAV  
        :param user: User whose task should be used 
        :type user: User 

        :return: 1: task did not expire | 0: task expired
        :rtype: int
        """
        # get task 
        task: Task = user.getTask(0)
        # get task time tolerance
        tolerance = task.tolerance
        if sp.writeOutput:
            print('Task tolerance: ', tolerance)
        # get task processing time cost 
        totalTime = sysModel.totalTime(uav, user)
        if sp.writeOutput:
            print('Task processing time from sys models: ', totalTime)
        # Calculate the time steps elapsed since the task was created
        elapsedSteps = self.timeSlot - task.creation
        elapsedTime = elapsedSteps * (sp.T/sp.N)
        # Add the task processing time to the time elapsed since the task was added to the queue 
        totalTime = elapsedTime
        if sp.writeOutput:
            print('Task processing total time with elapsed time: ', totalTime)
        # check if task expired or not 
        if totalTime <= tolerance:
            return 1
        else:    
            return 0

    # function to create a new environment state
    def newState(self, user: User, uav: UAV):
        """Crete a new environment state

        :param user: User's whose parameters should be used
        :type user: User 
        :param UAV: UAV whose parameters should be used 
        :type UAV: UAV 

        :return: The created state
        :rtype: Tensor
        """
        # set new state
        newState = State(uav, self.timeSlot, user)
        # normalize the state values
        newState.toTensor()
        # add new state to the list
        self.states.append(newState)
        return newState
    
    def newAction(self, uav: UAV, taskOffload, positionDelta, user = None):
        """Create a new action based on the selected action of the network

        :param uav: UAV that will use the position delta variable 
        :type uav: UAV 
        :param user: User that will use the task offloading variable
        :type user: User 
        :param taskOffload: Percentage of the task to be offloaded 
        :type taskOffload: float 
        :param positionDelta: Next movement of an UAV
        :type positionDelta: numpy array 
        """
        # create new action object 
        self.action = Action(taskOffload, positionDelta)
        # set uav future position
        uav.setpositionDelta(positionDelta)
        if user != None and user.tasks:
            # set task offload for the given user 
            user.setTaskOffload(taskOffload)

    def getAction(self):
        """Getter function for the action attribute

        :return: The current action
        :rtype: Action 
        """
        return self.action
        
    def reward(self, uav: UAV, tag, user: User = None): 
        """Calculate the reward the agent will receive for executing a given 
        action

        :param uav: UAV whose action will be rewarded 
        :type uav: UAV 
        :param user: User whose task and parameters will be used, defaults to None
        :type user: User 

        :return: The calculated value of the reward
        :rtype: float 
        """
        if user != None and self.tag.completed:
            task: Task = user.getTask(0)
            # check if task was processed before it expired
            expiration = self.checkExpiration(uav, user)
            term1 = (expiration*sp.lambda_1 * task.dataSize)/sp.taskDataSizeMin
            # energy consumption term 
            totalEnergy = sysModel.totalEnergy(uav, user, True)
            term2 = math.log(totalEnergy, 5)*sp.lambda_2
            # time consumption term 
            totalTime = sysModel.totalTime(uav, user, True)
            term3 = math.log(totalTime, 3)
            term3 = sp.lambda_3*term3
            # reward calculation
            reward = term1 - term2 - term3 + sp.constant
        elif user == None:
            expiration = 0
            # energy spent on movement 
            totalEnergy = sysModel.propulsionEnergy(uav)
            term2 = math.log(totalEnergy, 5)*sp.lambda_2
            # time consumption term 
            totalTime = 0
            # reward calculation
            reward = expiration - term2 - totalTime + sp.constant  
        elif self.tag.expired:
            # Task expired
            term1 = 0
            # Consider energy spent on that time step's movement
            if tag.elapsedSteps > 1:
                totalEnergy = sysModel.totalEnergy(uav, user, True)
            else:
                totalEnergy = sysModel.propulsionEnergy(uav)
            term2 = math.log(totalEnergy, 5)
            term2 = sp.lambda_2*term2
            # Time since task creation
            totalTime = (self.timeSlot - self.tag.task.creation)*(sp.T/sp.N)
            term3 = math.log(totalTime, 3)
            term3 = term3*sp.lambda_3
            # reward calculation
            reward = (term1 - term2 - term3) * sp.expirationPenalty
        elif not self.tag.completed and not self.tag.expired:
            # if task still being processed, reward is the small constant value 
            totalTime = totalEnergy = 0
            reward = sp.constant
        # Penalize the uav if it moves outside of the map bounds
        if uav.outOfBounds():
            reward += sp.penalty*uav.boundDistance()
        return reward
    
    def reset(self, uav : UAV, randomize = False):
        """Reset the  MEC environment by resetting the users, the UAV, the time step and the tag

        :param uav: UAV to be reset
        :type uav: UAV 
        :param randomize: Indicates whether the user positions should be ranzomized, defaults to False
        :type randomize: boolean
        """ 
        # Reset all user's in the world
        if randomize:
            self.world.users = []
            match sp.mapType:
                case 'square':
                    bounds = [sp.minBound,sp.maxBound,sp.minBound,sp.maxBound]
                case 'rectangle':
                    bounds = [sp.x1,sp.x2,sp.y1,sp.y2]
            print('Randomizing users')
            self.world.spawn(sp.totalUsers, bounds)
        # Reset the tasks of all user's
        else:
            for user in self.world.users:
                user.reset()
        # reset uav queue, position delta and current position to initial position 
        uav.reset()
        # reset environment time step 
        self.timeSlot = 0
        # reset stored states list 
        self.states = []
        # reset the environment tag
        self.tag.reset()

    def __str__(self):
        """print the environment's current time slot and the current state parameters
        """
        return f"Current time slot: {self.timeSlot}, Current State: {self.states[len(self.states)-1]}"
    
class TaskTag:
    """ Class to represent the task currently being processed in the environment 

    :param uav: UAV whose queue will be used 
    :type uav: UAV  
    :param world: World where the users are stored 
    :type world: World      
    """

    def __init__(self, uav: UAV, world: World):
        self.uav = uav
        self.world = world
        self.completed = False
        self.expired = False
        self.elapsedSteps = 0
        # task tracking 
        self.remaining = 0
        self.uploaded = 0
        self.offSize = 0
        self.calculate = True
        self.uploadTime = 0
        # expiration tracking
        self.timeStep = 0
        # reward calculation tracking
        self.lastCompleted = 0
        self.start = 0
        # time tracking 
        self.totalDuration = 0

    def checkExpired(self, currentStep: int, task: Task):
        """Check if the current task has already expired 

        :param currentTime: Current time step in the environment
        :type currentTime: int
        :param task: Task to be processed
        :type task: Task
        :return: True if expired, False if not 
        :rtype: boolean
        """
        # Check if task has exceeded its time limit 
        stepsDelta = currentStep - task.creation
        # Convert steps delta to seconds
        timeDelta = stepsDelta*(sp.T/sp.N)
        if(timeDelta > task.tolerance):
            if sp.writeOutput:
                print('Current timestep: ' + str(currentStep) + ' TASK EXPIRED: ', task)
            # task expired
            self.expired = True
            return True
        else:
            # can be processed 
            if sp.writeOutput:
                print('Current timestep: ' + str(currentStep) + ' TASK PROCESSING: ', task)
            return False

    def updateTask(self, task: Task):
        """Update the current task being processed

        :param task: Task to be processed
        :type task: Task  
        """
        # Store the current task 
        self.task: Task = task 
        # Check if task has expired 
        self.checkExpired(self.timeStep, self.task)
        # Update tag 
        self.start = self.timeStep
        
    def getDuration(self):
        """Calculate the time it will take to offload the task to the UAV based on the transmission rate, updated per step until it is finished.
        After that, check the total time cost of the local and offloaded computations to determine the total time cost.
        """
        # Check if there is a new task 
        if(self.offSize == 0 and self.task.d != None):
            # Get the user ID
            user = self.world.userFromID(self.task.userID)
            self.task = user.getTask(0)
            self.offSize = self.task.d * self.task.dataSize
            self.remaining = self.offSize
        # Check if all the data has been offloaded 
        if(self.remaining <= 0 and self.calculate):
            # Get the user ID
            user = self.world.userFromID(self.task.userID)
            # Calculate the total time the task will take to complete after the offloading 
            duration = sysModel.totalTimeAfterUpload(self.uav, user, self.uploadTime)
            # Update the UAV communication time for reward calculatio
            self.uav.transTime = self.uploadTime
            # Get the number of time steps to complete the task 
            self.elapsedSteps = 0
            self.stepstoComplete = utils.timeToSteps(duration)
            # Dont redo this calculation 
            self.calculate = False
            if sp.writeOutput:
                print('total upload time: ', self.uploadTime)
                print('steps to complete after upload: ', self.stepstoComplete)
        # Keep offloading the data at the current transmission rate
        elif(self.remaining > 0):
            # Get the user ID
            user = self.world.userFromID(self.task.userID)
            # Check how much data can be uploaded in the current time step
            uploadRate = sysModel.transmissionRate(self.uav, user)
            # Get time step duration
            timeStep = sp.T/sp.N
            # Calculate the amount of data remaining after the upload 
            self.uploaded = uploadRate*timeStep
            self.remaining -= self.uploaded
            # Get the number of time steps to complete the task  
            self.stepstoComplete = utils.timeToSteps(self.offSize/uploadRate) + 2 
            # Keep track of the total offloading time 
            if(self.remaining > 0):
                self.uploadTime += self.remaining/uploadRate
            else:
                self.uploadTime += (self.uploaded - abs(self.remaining))/uploadRate
                self.getDuration()
            if sp.writeOutput:
                print('uploaded: ', self.uploaded, 'remaining: ', self.remaining)
                print('steps to complete at current rate (+2): ', self.stepstoComplete)
            
    def incrementSteps(self):
        """Increment elapsed time steps since task was started and check if it is completed 

        :return: 1: task was completed | 0: task was not completed
        :rtype: Literal[1,0]
        """
        self.elapsedSteps += 1
        # Check task completion
        if sp.writeOutput:
            print('Elapsed steps: ' + str(self.elapsedSteps) + ' To complete: ' + str(self.stepstoComplete))
        if self.elapsedSteps == self.stepstoComplete:
            self.completed = True
            self.lastCompleted = self.timeStep
            return 1 
        else:
            # Check if task has expired 
            expired = self.checkExpired(self.elapsedSteps, self.task)
            if expired:
                return 1 
            if sp.writeOutput:
                print('TASK NOT COMPLETED')
            return 0 

    def reset(self):
        """Reset the task related attributes to None
        """
        self.task = None
        self.stepstoComplete = 0
        self.elapsedSteps = 0
        self.completed = False
        # task tracking 
        self.remaining = 0
        self.uploaded = 0
        self.offSize = 0
        self.calculate = True
        self.uploadTime = 0
        self.expired = False

    def __str__(self):
        """print the tag's task, steps to complete and elapsed steps 
        """
        return f"Task: {self.task}, Steps to complete: {self.stepstoComplete}, Elapsed Steps: {self.elapsedSteps}"