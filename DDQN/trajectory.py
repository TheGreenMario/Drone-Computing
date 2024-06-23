from abc import ABC, abstractmethod
import math 
import numpy as np
import simParam as sp
import itertools
import Utilities.utils as utils
import heapq
import hashlib

class actionAbs(ABC):

    @abstractmethod
    def random(self):
        pass
    
    @abstractmethod
    def createSpace(self):
        pass

    @abstractmethod
    def fromIndex(self, index):
        pass
    

class trjAction(actionAbs):
    """
    Select the next position to move to from N possible positions
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
            self.random(len(self.actionSpace))
        
    def createSpace(self):
        """Create a list with all possible task offloading and movement combinations to form the action space
        """
        # task offloading space
        d = sp.taskOffloading
        # movement action space 
        actionSpace = []
        if sp.generated:
            # Get the number of points the UAV can move to
            nbPoints = sp.nbPoints
            den = math.sqrt(nbPoints)
            # Check for invalid selection
            if not (den).is_integer():
                raise Exception("Selected number of points must have a valid square root")
            # Create the possible values for x and y 
            den = int(den)
            # Get the positions for the UAV in the range [0-1] 
            pointSet = np.linspace(1/(den*2), 1-(1/(den*2)), num=den)
            # Scale the value of the positions to the range of the map
            x = np.multiply(pointSet, sp.maxBound)
            # Round the value of the elements to 2 decimal cases
            for element in range(len(x)):
                x[element] = utils.roundHalfUp(x[element], 0)
            # Create the movement space with the (x,y) values
            y = x
            iterables = [x , y]
            movementSpace = []
            # cartesian product of the x and y values
            for t in itertools.product(*iterables):
                movementSpace.append(t)
            # Create the action space with the movement and offloading values
            iterables = [d , movementSpace]
        # user defined points
        else:
            iterables = [d , sp.validPoints]
        for t in itertools.product(*iterables):
            actionSpace.append(t)
        # set action space 
        self.actionSpace = actionSpace

    def random(self, totalActions):
        """Get a random action by generating a random index and seeing what it corresponds to on the action space list 

        :param totalActions: Number of total actions the agent can choose from
        :type totalActions: int
        """
        # generate a random index from the action space 
        actionIndex = utils.getValueInRange(0, totalActions-1)
        # get the corresponding action
        action = self.fromIndex(actionIndex)
        self.index = actionIndex
        return action

    def fromIndex(self, index):
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
    
class node():
    """Class to represent a node to be used in the A* algorithm

    :param parent: Defines the parent nodes used to reach the current node
    :type parent: node
    :param position: (x,y) position of the current node 
    :type position: list(int, int)
    :param g: Distance between the current node and the start node
    :type g: float 
    :param h: Heuristic function to estimate the distance between the current node and the end node 
    :type h: float  
    :param f: Total cost of the node 
    :type f: float 
    """

    def __init__(self, parent, position):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        """Defining equal operation for heap queue 
        """
        return self.position == other.position
    
    def __lt__(self, other):
      """Defining less than operation for heap queue 
      """
      return self.f < other.f
    
    def __gt__(self, other):
      """Defining greather than operation for the heap queue 
      """
      return self.f > other.f
    
    def __hash__(self):
        """Defining a hash value integer return for the set operation
        """
        id = hashlib.md5(f"{self.position[0]}_{self.position[1]}".encode()).hexdigest()
        return hash(id)

    def __str__(self):
      """Utility function to print the node's parameters 
      """
      return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

class trajectory():
    """Class to implement an A* algorithm to calculate the path between two positions
    (no movement validation is currently implemented, it is considered that the UAV can move to any point in range)
    """

    def __init__(self, start, end):
        self.start = start
        self.end = end 
        self.path = None
        self.move = 0
        self.moves()
        self.aStar((start[0], start[1]), (int(end[0]), int(end[1])))
        
    def moves(self):
        """Calculate how many moves the UAV can take per time step
        """
        steps = utils.distancePerStep()
        self.steps = steps

    def update(self):
        """Get the next position of the UAV by selecting it from the path positions according to the number of moves per time step 

        :return: UAV position
        :rtype: list
        """
        position = self.move + self.steps
        if position <= len(self.path) - 1:
            return self.path[int(position)]
        else:
            return self.path[int( len(self.path) - 1)]

    def getChildren(self, currentNode):
        """Function to calculate all the children of a given node 

        :param currentNode: Node whose children will be calculated
        :type currentNode: node

        :return: List of children of the given node
        :rtype: list[node]
        """
        # Inialize list to store the children
        children = []
        # Set the adjacency values for the children calculation 
        adjX = [-1, 0, 1]
        adjY = adjX
        adj = []
        # Get the list of adjacent nodes 
        iterables = [adjX , adjY]
        for t in itertools.product(*iterables):
            adj.append(t)
        # Get the position of each of the 8 possible children 
        for element in adj:
            # Get position
            position = (element[0] + currentNode.position[0], element[1] + currentNode.position[1])
            child = node(currentNode, position)
            # Add it to the list 
            children.append(child)
        return children

    def aStar(self, startPos, endPos):
        """Implement the A* algorithm to find the shortest path to the goal point 

        :param startPos: (x,y) pair to represent the initial position
        :type startPos: list(int, int)
        :param endPos: (x,y) pair to represent the end position
        :type endPos: list(int, int)

        :return path: list of positions that represent the trajectory or None if path was not found 
        :rtype: list((x,y)) | None
        """
        # initialize open and closed lists
        open = []
        heapq.heapify(open)
        closed = set()
        # Create start and end nodes 
        start = node(None, startPos)
        end = node(None, endPos)
        # Add start node
        heapq.heappush(open , start)
        niter = 0
        # Loop until the path is found 
        while open:
            # Get current node 
            currentNode = heapq.heappop(open)
            closed.add(currentNode)
            # If current node is the goal 
            if currentNode == end:
                path = []
                # Get reversed path
                while currentNode is not None:
                    path.append(currentNode.position)
                    currentNode = currentNode.parent
                self.path = path[::-1] 
                break
            # Get children 
            children = self.getChildren(currentNode)
            for child in children:
                # If child is already in the closed list, continue 
                if child in closed:
                    continue
                # Update values of the child 
                child.g = currentNode.g + 1 
                child.h = ((child.position[0] - end.position[0]) ** 2) + ((child.position[1] - end.position[1]) ** 2)
                child.f = child.g + child.h 
                # Child is already in the open list
                if child in open: 
                    idx = open.index(child) 
                    if child.g < open[idx].g:
                        # update the node in the open list
                        open[idx].g = child.g
                        open[idx].f = child.f
                        open[idx].h = child.h
                else:
                    # Add the child to the open list
                    heapq.heappush(open, child)
            niter = niter + 1 
            # If number of tries exceeds limit 
            if niter > sp.maxBound**2:
                return None
            
    def __str__(self):
      """Utility function to print the trajectory's parameters 
      """
      return f"Trajectory - Start: {self.start} End: {self.end}"