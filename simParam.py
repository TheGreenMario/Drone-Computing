"""
Module that includes most simulation parameters that can be tweaked to change the results accordingly

- UAV parameters
- System Model parameters
- MDP parameters
- DDQN parameters

----
"""
# Write output to terminal for debugging 
writeOutput = False

nbEpisodes = 100
"""Number of episodes to wait between saving episodic metrics: 100
"""
taskThreshold = 20
"""Task threshold for generating a new task: 50
"""

taskOffloading = [0.2, 0.4, 0.6, 0.8, 1]
"""Action space for the task offloading parameter
"""
movementOptions = [[-1,0],[-1,1],[-1,-1],[0,0],[0,1],[0,-1],[1,0],[1,1],[1,-1]]
"""Action space for the movement parameter
"""

mapType = 'square'
""" Define which map type to use: rectangle or square 
"""
# coordinate pairs for the rectangle map 
x1 = 0
""" Minimum x coordinate for the rectangle: 0 
"""
x2 = 100
""" Maximum x coordinate for the rectangle: 400
"""
y1 = 0
""" Minimum y coordinate for the rectangle: 0
"""
y2 = 2400
""" Maximum y coordinate for the rectangle: 2400
"""
# (x,y) bounds for the square map (in m)
minBound = 1
"""Lower bound for the map: 1 (m)
"""
maxBound = 1800
"""Upper bound for the map: 1800 (m)
"""
totalUsers = 100
"""Total number of users: 100
"""
randUsers = True
"""Whether the user's locations should be randomized every episode: True
"""
bound = False
"""Whether the movements of the UAV should be limited to the defined map bounds
"""
penalty = 0
""" UAV reward penalty if it is out of map bounds: 0
    Final penalty will be calculated by: penalty * distance to bounds
"""
expirationPenalty = 0.25
""" Float to multiply the value of the penalty for expired tasks 
"""
finalBounds = False
""" Whether the episode should end if the UAV leaves the map bounds 
"""
outOfBoundsPenalty = 100000
""" Penalty for going out of bounds 
"""
### System Model Variables ###
T = 3600
"""Total time: 3600 (in seconds)
"""
N = 720
"""Number or time slots:  3600
"""
epDuration = (10*60) + 1
""" Number of timesteps per episode
"""
totalEpisodes = 10000
""" Total number of episodes
"""

## UAV 
uavX = 2600
"""UAV spawn x coordinate
"""
uavY = maxBound/2
"""UAV spawn y coordinate
"""
uavH = 25
"""UAV flight altitude: 25 m
"""
M = 1000
"""Mass of (UAV + payload): 1e3 g
"""
uavMaxSpeed = 10
"""Maximum speed of the UAV: 10 m/s
"""
hovering = 1000
"""Consumption of the UAV when hovering in position: 1000
"""
### MoveTo settings
nbPoints = 2
"""The number of points the UAV can move to: 2
"""
generated = False
"""Wether the valid points are automatically generated or user defined: True
"""
validPoints = [
    [maxBound/2, 2400],
    [maxBound/2, maxBound/2],
]
"""List of user defined points that the UAV can move to
"""

## Communication Model
alpha = 9.61
"""Environment related variable: 9.61
"""
beta = 0.16
"""Environment related variable: 0.16
"""
fc = 2000
"""Carrier frequency: 2e3 hz
"""
c = 3*(10**8)
"""Light speed constant: 3e8 m/s
"""
nLoS = 1.0
"""Line of Sight path loss variable: 1.0
"""
nNLoS = 20
"""Non Line of Sight path loss variable: 20.0
"""
bandwidth = 10**7
"""bandiwdth value: 10e7 hz
"""
power = 36
"""Transmission power of a mobile user value: 36 W
"""
nZero = -174
"""Value of the noise spectral density: -174 dBm
"""
fUAV = 32*(10**9)
"""CPU frequency of the UAV: 32e9 hz
"""
hwdConst = 10**(-26)
"""Hardware related constant: 10e-26
""" 

### Markov Decision Process Variables ###

## State Space

# user device CPU frequency range in hz 
fUserMin = 8*(10**8)
"""User device CPU min frequency: 8e8 hz
"""
fUserMax = 8*(10**9)
"""User device CPU max frequency: 8e9 hz 
"""
hUAV = 25
"""Initial UAV altitude: 25 m
"""
maxQueueTasks = 500
"""Max number of tasks in the queue: 500
"""
queueExpiration = 10 
"""Queue expiration time: 10 s
"""
# task data size in bits 
taskDataSizeMin = 2*(10**7)
"""Min task data size: 2e7 bits
"""
taskDataSizeMax = 2*(10**8)
"""Max task data size: 2e8 bits
"""
# required CPU cycles to calculate per bit data
cpuCyclesMin = 5
"""Min CPU cycles: 5
"""
cpuCyclesMax = 6
"""Max CPU cycles: 6 
"""
# task tolerance time in seconds 
taskWaitTimeMin = 60 
"""Min task tolerance: 10 s
"""
taskWaitTimeMax = 120 
"""Max task tolerance: 30 s
"""
## Reward Function 

constant = 1e-1
"""Small constant to keep the model running and accumulate rewards over steps: 1e-2
""" 
gamma = 0.99
"""Discount rate: 0.99
"""
# weight values for each component
lambda_1 = 100 * (T/N)
"""Weight value for task completion: 10
"""
lambda_2 = 5
"""Weight value for energy consumption term: 10
"""
lambda_3 = 15
"""Weight value for time consumption term: 10
"""

### DDQN Variables ###

## Hyperparameters
memorySize = 100000
"""Buffer size for the experience replay buffer: 10000
"""  
bufferSize = 10**5
"""Buffer size: 10e5
"""
batchSize = 64
"""Batch size: 64
"""
epsilon = 0.99
"""Initial value for the epsilon: 0.99
"""
epsilonMin = 1e-3
"""Min epsilon: 1e-3
"""
epsilonDecay = 0.99995
"""Epsilon decay: 0.99995
"""
learningRate = 1e-4
"""Learning rate: 1e-4
"""
learnStep = 10
""" Number of steps between updates to the online network: 10
"""
tau = 1e-4
"""Soft update discount: 1e-4
"""