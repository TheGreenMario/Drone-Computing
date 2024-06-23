"""
Module used to represent the replay memory, qAgent and qNetwork classes

----
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import simParam as sp
import MEC.MDP as mdp
from .trajectory import trjAction

class memoryBuffer(object):
    """Class to define the memory buffer using numpy arrays

    :param size: Buffer size (equal to number of time slots N)
    :type size: int
    :para memCounter: Keep track of the number of memories in the buffer
    :type memCounter: int
    :param currentState: Buffer to store the current MDP state
    :type currentState: numpy array
    :param nextState: Buffer to store the next MDP state
    :type nextState: numpy array
    :param action: Buffer to store the current MDP action
    :type action: numpy array
    :param reward: Buffer to store the current reward
    :type reward: numpy array
    """
    def __init__(self, stateSize):
        """Constructor to initialize the memory buffer class
        """
        # buffer size
        self.size = sp.memorySize
        # keep track of the number of memories in the buffer
        self.memCounter = 0
        # create the memory buffer
        self.currentState = np.zeros((self.size, stateSize),
                                     dtype=float)
        self.nextState = np.zeros((self.size, stateSize),
                                         dtype=float)

        self.action = np.zeros(self.size, dtype=np.int64)
        self.reward = np.zeros(self.size, dtype=float)
        self.done = np.zeros(self.size, dtype=bool)

    # function to store the given record in the memory buffer
    def storeMemory(self, currState, currAction, reward, nextState, done):
        """Function to store the given experience using numpy arrays and increment the memory counter

        :param currState: Current MDP state
        :type currState: Tensor
        :param currAction: Current MDP Action
        :type currAction: Tensor
        :param reward: Current reward based on the state and chosen action
        :type reward: Tensor
        :param nextState: Next MDP state after action is executed by the agent
        :type nextState: Tensor
        """
        # insert the new memory into the corresponding buffer position
        # Reset counter
        if self.memCounter == self.size:
            self.memCounter = 0
        self.currentState[self.memCounter] = currState
        self.nextState[self.memCounter] = nextState
        self.action[self.memCounter] = currAction
        self.reward[self.memCounter] = reward
        self.done[self.memCounter] = done
        # increment the total memories saved
        self.memCounter += 1

    def sample(self, batchSize):
        """Sample a batch of memories from the buffer

        :param batchSize: Value of the batch size hyperparameter (simulation parameters)
        :type batchSize: int

        :return: boolean to indicate if there were enough memories in the buffer, and experience buffers
        :rtype: list[boolean, memories (Tensor) or None]
        """
        # check if there are enough memories in the buffer 
        if self.memCounter-1 > batchSize:
            # generate a batch of unique indices from range 0-self.memCounter without replacement
            batch = np.random.choice(self.memCounter-1, batchSize, replace=False)
            states = self.currentState[batch]
            actions = self.action[batch]
            rewards = self.reward[batch]
            states_ = self.nextState[batch]
            done = self.done[batch]
            return [True, states, actions, rewards, states_, done]
        else:
            return [False, None]
        
class qNetwork(nn.Module):
    """Class to define the Q-network architecture

    :param hidden: linear hidden layers  
    :type hidden: Linear
    :param outputLayer: output linear layer
    :type outputLayer: Linear
    :param device: Define the device (GPU or CPU) to be used in torch operation
    :type device: Device
    """
    def __init__(self, stateSize, actionSize):
        """Constructor to initialize the q-network architecture
        """
        # calls __init__ method of nn.Module class
        super(qNetwork,self).__init__()
        # define the hidden layers
        self.hidden1 = nn.Linear(stateSize, 256, dtype=T.float64)
        self.hidden2 = nn.Linear(256, 256, dtype=T.float64)
        self.hidden3 = nn.Linear(256, 256, dtype=T.float64)
        self.hidden4 = nn.Linear(256, 256, dtype=T.float64)
        self.hidden5 = nn.Linear(256, 256, dtype=T.float64)
        # define the output layer
        self.outputLayer = nn.Linear(256, actionSize, dtype=float)
        # use a gpu if available 
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # ensure all operations use the same device (cpu or gpu)s
        self.to(self.device)


    def forward(self, state):
        """Called with either one state or a batch of states to perform a forward pass through the network and determine the next action

        :param state: State element or array of state elements 
        :type state: Tensor or numpy array of Tensors

        :return: Result of the output layer
        :rtype: Linear
        """
        # perform the forward pass through the various layers using the Rectified Linear Unit function
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        return self.outputLayer(x)

class qAgent(object):
    """Class to define the Q-Learning Agent

    :param stateSize: Size of the state 
    :type stateSize: int
    :param actionSize: Size of the action space
    :type actionSize: int
    :param bufferSize: Size of the buffer hyperparameter (from simulation parameters)
    :type bufferSize: int
    :param batchSize: Size of the batch hyperparameter (from simulation parameters)
    :type batchSize: int
    :param epsilonMin: Minimum value the epsilon can reach (from simulation parameters)
    :type epsilonMin: float 
    :param epsilonDecay: Value the epsilon decreases per time step (from simulation parameters)
    :type epsilonDecay: float 
    :param epsilon: Initial value of the epsilon
    :type epsilon: float
    :param gamma: Gamma value (from the simulation parameters)
    :type gamma: float
    :param lr: Learning rate for the Adam optimizer (from the simulation parameters)
    :type lr: float
    :param buffer: Memory replay buffer
    :type buffer: memoryBuffer class
    :param onlineNetwork: Online Q-Network
    :type onlineNetwork: qNetwork class
    :param targetNetwork: Target Q-Network 
    :type targetNetwork: qNetwork class
    :param optimizer: Optimizer for the online network with the given learning rate 
    :type optimizer: Adam class
    :param loss: Loss function to be applied
    :type loss: SmoothL1Loss class
    :param scoreWindow: Keep a record of the scores
    :type scoreWindow: list[float]
    :param episodeScore: Keep a record of all scores for an episode
    :type episodeScore: list[float]
    :param optimalValue: Optimal value for the Q-Network
    :type optimalValue: float
    """ 
    
    def __init__(self, stateSize, actionSize):
        """Constructor to initialize the agent
        """
        # define state and action sizes
        self.stateSize = stateSize
        self.actionSize = actionSize

        # define hyperparameters from the simulation parameters
        self.bufferSize = sp.bufferSize
        self.batchSize = sp.batchSize
        self.epsilonMin = sp.epsilonMin
        self.epsilonDecay = sp.epsilonDecay
        self.epsilon = sp.epsilon
        self.gamma = sp.gamma
        self.lr = sp.learningRate

        # initialize replay memory buffer
        self.buffer = memoryBuffer(stateSize)

        # create the online and target Q-network
        self.onlineNetwork = qNetwork(stateSize, actionSize)
        self.targetNetwork = qNetwork(stateSize, actionSize)
        # initialize the optimizer 
        self.optimizer = optim.Adam(self.onlineNetwork.parameters(), self.lr)
        # initialize loss function 
        self.loss = nn.SmoothL1Loss()

    def selectAction(self, state, force=False):
        """Select an action according to an epsilon greedy policy 

        :param state: State element or Array of State elements
        :type state: State
        :param force: True if an action should be random indenpendently of the epsilon value, defaults to false
        :type force: boolean

        :return: The randomly selected action or the best action
        :rtype: Action
        """
        # choose best action
        # np.random.random() returns a float in the interval [0,1)
        if np.random.random() > self.epsilon or force:
            # get actions by performing a forward pass in the network
            actions = self.onlineNetwork.forward((state.toTensor()).to(self.onlineNetwork.device))
            # get the best action
            actionIndex = T.argmax(actions).item()
            # select from action space 
            action = mdp.Action()
            action.fromIndex(actionIndex)
            if sp.writeOutput:
                print('Best action: ', action)
        # choose random action
        else:
            action = mdp.Action()
            if sp.writeOutput:
                print('Random action: ', action)

        return action
    
    def selectTrajAction(self, state, force=False):
        """Select an action according to an epsilon greedy policy 

        :param state: State element or Array of State elements
        :type state: State
        :param force: True if an action should be random indenpendently of the epsilon value, defaults to false
        :type force: boolean

        :return: The randomly selected action or the best action
        :rtype: Action
        """
        # choose best action
        # np.random.random() returns a float in the interval [0,1)
        if np.random.random() > self.epsilon or force:
            # get actions by performing a forward pass in the network
            actions = self.onlineNetwork.forward((state.toTensor()).to(self.onlineNetwork.device))
            # get the best action
            actionIndex = T.argmax(actions).item()
            # select from action space 
            action = trjAction()
            action.fromIndex(actionIndex)
            if sp.writeOutput:
                print('Best action: ', action)
        # choose random action
        else:
            action = trjAction()
            if sp.writeOutput:
                print('Random action: ', action)
    
        return action
    
    def storeMemory(self, currState, currAction, reward, nextState, done):
        """Add an experience to the memory buffer 

        :param currState: Current MDP state
        :type currState: Tensor
        :param currAction: Current MDP Action
        :type currAction: Tensor
        :param reward: Current reward based on the state and chosen action
        :type reward: Tensor
        :param nextState: Next MDP state after action is executed by the agent
        :type nextState: Tensor
        """
        self.buffer.storeMemory(currState, currAction, reward, nextState, done)
    
    def sampleMemory(self):
        """Select a batch of states from the memory buffer

        :return: Batch of memories from the buffer
        :rtype: list[boolean, memories or None]
        """
        batch = self.buffer.sample(self.batchSize)
        return batch
    
    def decrementEpsilon(self):
        """Decrement the current epsilon by the decay value
        """
        #self.epsilon = self.epsilon*self.epsilonDecay
        self.epsilon *= sp.epsilonDecay
        # Check if the new value is valid
        self.epsilon = max(self.epsilonMin, self.epsilon)
        
    def td_estimate(self, state, action): 
        """Predicted optimal Q value for a given state

        :param state: State array to be passed through the network
        :type state: Numpy array of state tensors
        :param action: Action element to be used in the forward pass
        :type action: Numpy array of action tensors

        :return: Optimal Q for a given state
        :rtype: float
        """ 
        current_Q = self.onlineNetwork.forward(T.tensor(state).to(self.onlineNetwork.device))[np.arange(0, self.batchSize), action]
        return current_Q

    @T.no_grad()
    def td_target(self, reward, nextState, done):
        """Aggregation of current reward and predicted Q value of the next state 

        :param reward: Reward of the current state action pair 
        :type reward: float 
        :param nextState: State array to be passed through the network
        :type nextState: State

        :return: Predicted reward of the current state and the next state 
        :rtype: float
        """
        next_state_Q = self.onlineNetwork.forward(T.tensor(nextState).to(self.onlineNetwork.device))
        best_action = T.argmax(next_state_Q, axis=1)
        next_Q = self.targetNetwork.forward(T.tensor(nextState).to(self.targetNetwork.device))[np.arange(0, self.batchSize), best_action]
        return (T.tensor(reward).to(self.onlineNetwork.device) + (1 - T.tensor(done).to(self.onlineNetwork.device).float() ) * self.gamma * next_Q)
    
    def updateOnlineQ(self, td_estimate, td_target):
        """Update the online Q-Network

        :param td_estimate: Optimal Q value for a given state
        :type td_estimate: float
        :param td_target: Predicted Q value for the current and next state
        :type td_target: float
        
        :return: Loss backpropagated through the online network
        :rtype: float 
        """
        loss = self.loss(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def softUpdate(self, tau):
        """Execute the soft-update to the target network 

        :param tau: Soft-update discount value
        :type tau: float
        """
        for target_param,online_param in zip(self.targetNetwork.parameters(),self.onlineNetwork.parameters()):
            target_param.data.copy_(tau*online_param.data + (1.0-tau)*target_param.data)