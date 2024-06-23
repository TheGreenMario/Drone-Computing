"""
Module to represent the User, UAV, Task and World as classes

----
"""
import simParam as sp
import numpy as np
import hashlib
import Utilities.utils as utils
import math 

# base class for both UAV and User classes
class Agent:
    """Class to represent agents that participate in the world 
    """

    def getX(self):
        """ Return the [x] value of a 3D cartesian position described by a numpy array 

        :return: x position coordinate
        :rtype: int
        """
        return self.position[0]
    
    def getY(self):
        """ Return the [y] value of a 3D cartesian position described by a numpy array 

        :return: y position coordinate
        :rtype: int
        """
        return self.position[1]
    
    def getH(self):
        """ Return the [z] value of a 3D cartesian position described by a numpy array 

        :return: z position coordinate
        :rtype: int
        """
        return self.position[2]

class UAV(Agent):
    """A class used to represent a UAV

    :param position: 3D cartesian position of the UAV (height is constant)
    :type position: numpy array
    :param positionDelta: 3D array to represent the next movement of the UAV as chosen by the learning network
    :type positionDelta: numpy array 
    :param nextPosition: The next position of the UAV resulting from a sum of the current position and the position delta
    :type nextPosition: numpy array
    :param queue: A list to store the tasks queue of the UAV 
    :type queue: list[Task]
    :param alg: The algorithm to be used
    :type alg: string
    :param propEnergy: Propulsion energy spent so far 
    :type propEnergy: float
    :param transTime: Total transmission time 
    :type transTime: float
    """

    def __init__(self, algorithm):
        """Constructor method 
        """
        # initialize position from simulation parameters 
        self.position = np.array([sp.uavX, sp.uavY, sp.uavH], dtype='int')
        self.positionDelta = None
        self.nextPosition = None
        # initialize queue for tasks 
        self.queue = []
        # specify the algorithm being used
        self.alg = algorithm
        # propulsion energy 
        self.propEnergy = 0
        # transmission time 
        self.transTime = 0

    def addTask(self, task): 
        """ Add a task instance to the UAV queue

        :param task: A task to be added
        :type task: Task 

        """
        # put item into queue   
        self.queue.append(task)

    def getTopTask(self):
        """ Get the top task in the queue and remove it 

        :return: Task  if sucessfull, None if queue is empty 
        :rtype: Task or None
        """
        try:
            # remove and return the top item from the queue 
            task = self.queue.pop(0)
            return task
        except:
            # no tasks left in the queue
            if sp.writeOutput:
                print('No tasks in the queue')
            return None
    
    def getAllTasks(self):
        """ Get all tasks in the queue without removing any

        :return: All tasks
        :rtype: list[Task]
        """
        return self.queue
    
    def setpositionDelta(self, positionDelta):
        """ Set the positionDelta variable and calculate the next position

        :param positionDelta: 3D array to represent the next movement of the UAV as chosen by the learning network
        :type positionDelta: numpy array 
        """
        self.positionDelta = positionDelta
        # If the moveToLocation algorithm is being used 
        if self.alg == 'moveTo':
            self.calculateNextTrajPosition()
        else:
            self.calculateNextPosition()

    def calculateNextPosition(self):
        """ Calculate the UAV's next position based on the maximum velocity possible and check if it is within bounds. If it is out of bounds,
        keep the current position. Set the UAV's nextPosition variable to the result.
        (Currently removed out of bounds checks)
        """
        # Calculate the movement the UAV performs in one time step
        movement = self.positionDelta*(sp.T/sp.N)*sp.uavMaxSpeed
        # Update the (x,y) pair with the new values 
        nextX = self.position[0] + movement[0]
        nextY = self.position[1] + movement[1]
        # limit the UAV position to the bounds of the map  
        if sp.bound:
            if nextX < sp.minBound or nextX > sp.maxBound:
                nextX = self.position[0]
            if nextY < sp.minBound or nextY > sp.maxBound:
                nextY = self.position[1]
        # define the UAV's next position
        self.nextPosition = np.array([nextX, nextY, sp.uavH], dtype='int')

    def updatePosition(self):
        """Update the value of the UAV's position variable to the value of its nextPosition variable
        """
        # move the UAV
        self.position = self.nextPosition
        # If the moveToLocation algorithm is being used 
        if self.alg == 'moveTo':
            if self.traj.move >= len(self.traj.path) - 1:
                self.traj = None

    def calculateNextTrajPosition(self):  
        """Next position to be taken from the trajectory path list  
        """
        position = self.traj.update()
        if position:
            self.nextPosition = np.array([position[0], position[1], sp.uavH], dtype='int')
       
    def setPath(self, trajectory):
        self.traj = trajectory
        
    def reset(self):
        """Reset the UAV agent by reseting its position to the initial default, setting the positionDelta to None and 
        making the queue an empty list.
        """
        # reset position delta and uav task queue 
        self.positionDelta = None
        self.queue = []
        self.nextPosition = None
        # reset uav position to its initial one
        self.position = np.array([sp.uavX, sp.uavY, sp.uavH])

    def outOfBounds(self):
        """Check if the UAV is out of bounds in the current position

        :return: True if out of bounds, False otherwise 
        :rtype: bool
        """
        match sp.mapType:
            case 'square':
                bounds = [sp.minBound, sp.maxBound, sp.minBound, sp.maxBound]
            case 'rectangle':
                bounds = [sp.x1, sp.x2, sp.y1, sp.y2]
        if(self.getX() > bounds[1]
           or self.getX() < bounds[0] 
           or self.getY() > bounds[3] 
           or self.getY() < bounds[2]):
            return True
        else:
            return False
        
    def boundDistance(self):
        """Get the distance the UAV is from the map bounds 

        :return: Distance 
        :rtype: float 
        """
        # Get the center of the square
        center = np.array([sp.maxBound/2, sp.maxBound/2])
        position = np.array([self.getX(), self.getY()])
        # Calculate distance to the center 
        distance = math.dist(center, position)
        return distance

    def __str__(self):
        """print the UAV's current position, position delta and total queue size
        """
        return f"Current Position: {self.position}, Position Delta: {self.positionDelta}, Tasks in Queue: {len(self.getAllTasks())}"
        

class User(Agent):
    """A class used to represent a user

    :param id: A unique user ID 
    :type id: int 
    :param cpuFreq: The user's CPU frequency generated according to the simulation parameters
    :type cpuFreq: int
    :param position: The position of the generated user
    :type position: numpy array 
    :param tasks: The tasks the user has to offload to the UAV
    :type task: Task[]
    :param processed: Number of tasks sucessfully processed
    :type processed: float 
    :param expired: Number of tasks that expired 
    :type expired: float
    :param generated: Total tasks generated for this user
    :type generated: float 
    """

    def __init__(self, x, y, id = 0):
        """Constructor method to initialize the instance with a given (x,y) pair of coordinates 
        """
        # define unique user ID 
        self.id = id
        # get the user cpu frequency from the simulation parameters 
        self.cpuFreq = utils.getValueInRange(sp.fUserMin, sp.fUserMax)
        # define user location 
        self.position = np.array([x, y, 0])
        # Create list to store tasks  
        self.tasks = []
        # Keep track of how many tasks were processed
        self.processed = 0
        # Keep track of how many tasks expired
        self.expired = 0
        # Keep track of how many tasks the user generated
        self.generated = 0

    def createTask(self, timeStep):
        """Create a task instance, with a random data size, and associate it to the the user's task list 

        :param timeStep: Timestep at which the task was generated
        :type timeStep: int 
        """
        # get total task data size from simulation values
        taskSize = utils.getValueInRange(sp.taskDataSizeMin, sp.taskDataSizeMax)
        # create the task with that value 
        self.tasks.append(Task(taskSize, self.id, timeStep))
        # increment tasks generated 
        self.generated += 1 
    
    def setTaskOffload(self, taskOffload):
        """Set the task offload percentage

        :param taskOffload: Percentage of the task to be offloaded to the UAV (0-1)
        :type taskOffload: float
        """
        # set task offload of the task at the top of the queue 
        self.tasks[0].setOffload(taskOffload)

    def getTask(self, taskNb):
        """Get the user's task at the top of the queue 

        :return: The task associated with the user
        :rtype: Task
        """
        if self.tasks:
            return self.tasks[taskNb]
        else:
            return None
    
    def completed(self):
        """Remove the completed task from the user's queue, and increment tasks completed
        """
        self.tasks.pop(0)
        self.processed += 1

    def incomplete(self):
        """Remove the expired task from the user's queue, increment expired tasks
        """
        self.tasks.pop(0)
        self.expired += 1

    @property
    def notCompleted(self):
        """Calculate the difference between the number of tasks generated and processed

        :return: The number of tasks that were not processed
        :rtype: int 
        """
        return self.generated - self.processed
    
    def reset(self):
        """Reset the user's task's list to empty, set tasks processed as zero and maintain the same position 
        """
        # delete all tasks 
        if self.tasks:
            self.tasks = []
        self.processed = 0
        self.expired = 0
        
    def __str__(self):
        """Print the user's position, top task parameters and total number of tasks in its queue 
        """
        return f"X: {self.getX()}, Y: {self.getY()}, Z: 0, with Selected Task Parameters - {self.getTask(0)}, and {len(self.tasks)} total tasks. Tasks: completed- {self.processed} expired- {self.expired}"

class Task:
    """A class used to represent a task

    :param creation: Time step at which the task was generated
    :type creation: int
    :param dataSize: Total number of bits required to be processed to complete the task
    :type dataSize: int
    :param userID: ID of the user associated to the task
    :type userID: int
    :param d: Percentage of the task to be offloaded to the UAV (0-1)
    :type d: float
    :param cpuCycles: Number of CPU cycles needed to process complete the task 
    :type cpuCycles: int
    :param tolerance: Maximum time for the task to be completed before expiration
    :type tolerance: int
    """
    
    def __init__(self, dataSize, id, creation):
        """Constructor method to initialize the task

        :param dataSize: Total number of bits required to be processed to complete the task
        :type dataSize: int
        :param id: ID of the user associated to the task
        :type id: int
        :param normalized: Array with the normalized values of the task's parameters
        :type normalized: numpy array 
        """
        self.creation = creation
        self.dataSize = dataSize
        self.userID = id 
        self.d = None
        # get the number of cpu cycles from the simulation parameters 
        self.cpuCycles = utils.getValueInRange(sp.cpuCyclesMin, sp.cpuCyclesMax)
        # set the task tolerance from the simulation parameters
        self.tolerance = utils.getValueInRange(sp.taskWaitTimeMin, sp.taskWaitTimeMax)
    
    def setOffload(self, d):
        """Set the task's offloading percentage

        :param d: Task offloading percentage (0-1)
        :type d: float
        """
        self.d = d
    
    def getDataSize(self):
        """Get the task's data size

        :return: Task's data size
        :rtype: int
        """
        return self.dataSize
    
    def getOffload(self):
        """Get the task's offloading percentage

        :return: Task's offloading percentage
        :rtype: float
        """
        return self.d
    
    def __str__(self):
        """print the tasks's data size, data offloading percentage, user ID, CPU Cycles and time tolerance
        """
        return f"Data Size: {self.getDataSize()}, Data offload: {self.getOffload()}, UserID: {self.userID}, CpuCycles: {self.cpuCycles}, Tolerance: {self.tolerance}, Creation Timestep: {self.creation}"

class World:
    """A class used to represent the 3D cartesian square world 

    :param users: Store user instances
    :type users: list[User]
    :param userDict: User dictionary using ID as key to identify users that are associated with a given task
    :type userDict: dictionary{id:User}
    :param uav: UAV agent associated with the world
    :type uav: UAV class
    """
   
    def __init__(self):
        """Constructor method to initialize the world
        """
        # create list to store user instances 
        self.users = []
        # user dictionary with ID as key 
        self.userDict = {}
        # create var to store the uav agent
        self.uav = None
    
    def addUser(self, user : User):
        """Add a new user to the user list

        :param user: user to be added
        :type user: User
        """
        # append new agent to the list 
        self.users.append(user)
        self.userDict[user.id] = user
        
    def userFromID(self, id):
        """Get the user instace given the id from a task

        :param id: User's ID
        :type id: string 

        :return: The user that belongs to that id or None if they do not exist  
        :rtype: User  or None
        """
        if id in self.userDict:
            return self.userDict[id]
        else:
            return None

    def checkVacancy(self, position):
        """Check if the given position in the world is already occupied

        :param position: coordinates to be checked 
        :type position: numpy array

        :return: Return True if no user with those coordinates was found 
        :rtype: bool
        """
        # generate user ID for the given position
        id = self.generateID(position[0], position[1])
        # check if another user with that position has already been created 
        result = any(a.id == id for a in self.users)
        # return true if no user was found 
        return not result
    
    def generateAgent(self, agentType, bounds, location=None):
        """Create a new agent (UAV or User) with the given location. 
        If no location was given, one will be randomly generated based on the world's bounds and 
        positions already occupied by other users.

        :param agentType: Define if the agent to be generated is a User or a UAV
        :type agentType: string
        :param location: Initial position for the new agent, defaults to None
        :type location: numpy array 

        :return: Returns the generated agent if sucessfully generated and None otherwise
        :rtype: UAV or User or None
        """
        # if no location is given, randomly generate one within bounds 
        if location == None:
            vacancy = False
            # loop until random location is unoccupied 
            while not vacancy:
                # create a random location 
                x = utils.getValueInRange(bounds[0], bounds[1])
                y = utils.getValueInRange(bounds[2], bounds[3])
                location = np.array([x, y, 0])
                # check if no agent is occupying that location
                vacancy = self.checkVacancy(location)
            # check agent type
            if agentType == 'User':
                # create new agent with the location
                id = self.generateID(x, y)
                u = User(x, y, id)
                # add user to the list 
                self.addUser(u)
            elif agentType =='UAV':
                # create new uav with the location
                u = UAV(x, y, sp.hUAV)
                self.uav = u
            else:  
                print('Invalid agentType')
                return None
        # if location is given
        else:
            # check if it is within bounds 
            if location[0] >= sp.minBound and location[0] <= sp.maxBound and  location[1] >= sp.minBound and location[1] <= sp.maxBound:
                # if it is a valid location, create the agent 
                if agentType == 'User':
                    # create new agent with the location
                    id = self.generateID(location[0], location[1])
                    u = User(location[0], location[1], id)
                    # add user to the list 
                    self.addUser(u)
                elif agentType =='UAV':
                    # create new uav with the location
                    u = UAV(location[1], location[1], sp.hUAV)
                else:
                    return None
        # return created agent
        return u
                    
    def generateID(self, x ,y):
        """Generate a user id using the (x,y) coordinates applied to the MD5 hash function to create unique ID's

        :param x: x coordinate
        :type x: int
        :param y: y coordiante
        :type y: int

        :return: generated user id
        :rtype: string
        """
        id = hashlib.md5(f"{x}_{y}".encode()).hexdigest()
        return id
    
    def totalUsers(self):
        """Return the total number of users in the world

        :return: total number of users
        :rtype: int
        """
        # get the total users
        users = len(self.users)
        return users

    def spawn(self, nbUsers, bounds):
        """Spawn users in random locations of the world until it is fully populated 

        :param nbUsers: Number of users to be generated
        :type nbUsers: int 
        :param bounds: List of minimum and maximum values the x and y can take with the format List[xMinimum, xMaximum, yMinimum, yMaximum]
        :type bounds: List[int]
        """
        userCount = 0
        # populate the world with users
        while userCount < nbUsers:
            # create a user in a random location 
            self.generateAgent('User', bounds)
            userCount += 1

    def __str__(self):
        """print the world's bounds and total users
        """
        return f"Number of Users: {len(self.users)}"