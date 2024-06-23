"""
Module to implement the DDQN algorithm with waypoint movement.

In this case, the agent has a number of possible positions to move to in the map, defined in the simulation parameters, and the position delta 
is the selected location.The UAV will then move there by following a path determined by the A* algorithm.

---
"""
import DDQN.DDQN as ddqn
import MEC.mecNetwork as mec 
import MEC.MDP as mdp
import simParam as sp
import MEC.systemModel as model
from Utilities.logger import Logger, listLog
from MEC.tagManagement import preprocessTimeline, actionTimeline
import Utilities.utils as utils
from DDQN.trajectory import trajectory
import torch as T

print('-------- Initialization --------') 
# initialize the ddqn agent
agent = ddqn.qAgent(stateSize=10, actionSize=sp.nbPoints*len(sp.taskOffloading))
# initialize the environment 
environment = mdp.Environment()
environment.startWorld()
if sp.writeOutput:
    print('Generated world: ', environment.world)
# initialize the UAV 
uav = mec.UAV('waypoint')
# Add task tracking to environment 
environment.addTag(uav)
if sp.writeOutput:
    print('Generated UAV agent: ', uav)
# directory path to save the generated output metrics
savePath = utils.dirSetup("./output/waypoint/")
print('Output directory: ', savePath)
saveMetrics = True
# initialize metrics 
metrics = Logger(savePath)
if saveMetrics:
    stepsMetric = listLog(savePath, 'Time_Steps')
    rewardsMetric = listLog(savePath, 'Rewards')
    offloadMetric = listLog(savePath, 'Task_Offloading')
    completedMetric = listLog(savePath, 'Completed_Tasks')
    expiredMetric = listLog(savePath, 'Expired_Tasks')
    generatedMetric = listLog(savePath, 'Generated_Tasks')
    lossMetric = listLog(savePath, 'Loss')
try:
    # main loop
    for episode in range(sp.totalEpisodes):
        print('---------------------------------------------------------------- Episode ' + str(episode) + ' ----------------------------------------------------------------') # remove
        # reset the MEC environment
        environment.reset(uav, sp.randUsers)
        # initialize user variable
        user = None
        # termination state variable
        done = False
        # uav trajectory
        uav.setPath(None)
        # initial input raw data S1
        state = environment.newState(None, uav)
        # time step loop
        for timeStep in range(sp.epDuration): 
            if sp.writeOutput:
                print('-------- Timestep loop ' +  str(timeStep) + '--------') # remove 
            # Select action based on the current state using the epsilon greedy policy 
            if sp.writeOutput:
                print('Current User: ', user)
            action = agent.selectTrajAction(environment.states[timeStep])
            if uav.traj == None:
                uav.traj = trajectory(uav.position[:2], action.positionDelta)
            else:
                uav.traj.move += uav.traj.steps
                # keep the same movement action while the UAV has not reached the end position of the trajectory 
                action.positionDelta = uav.traj.end
            if saveMetrics:
                offloadMetric.addValue(action.taskOffload)
            # Check if a task is currently being processed 
            actionTimeline(environment, uav, action, user)
            if sp.writeOutput:
                print('Task Tag: ', environment.tag)
            # Get reward
            reward = environment.reward(uav, environment.tag, user)
            # Calculate UAV propulsion energy
            uav.propEnergy += model.propulsionEnergy(uav)
            # Next UAV Position
            uav.updatePosition()
            # Update density map every N episodes 
            if saveMetrics and (episode == 0 or episode%sp.nbEpisodes == 0):
                metrics.addPosition(uav.position)
            # Log new Position 
            if saveMetrics:
                metrics.addPosition(uav.position)
            gen = environment.generateTask(uav, timeStep)
            if gen == 1:
                generatedMetric.addValue(1)
            # End episode after <epDuration> time steps have elapsed 
            if(timeStep + 1 == sp.epDuration):
                done = True
            if saveMetrics:
                rewardsMetric.addValue(reward)
            if sp.writeOutput:
                print('Reward: ', reward)
            # If task was completed in this time step
            if environment.tag.completed or environment.tag.expired:
                # reset propulsion energy
                uav.propEnergy = 0
                # Remove task from the user's list of tasks 
                if(environment.tag.completed):
                    user.completed()
                    if saveMetrics:
                        completedMetric.addValue(1)
                else:
                    user.incomplete()
                    if saveMetrics:
                        expiredMetric.addValue(1)
                # Reset the tag 
                environment.tag.reset()
            # Preprocess next state
            user = preprocessTimeline(environment, uav)
            # Create the next state 
            nextState = environment.newState(user, uav)
            # Save next state
            environment.states[timeStep+1] = nextState
            if sp.writeOutput:
                print('Next State: ', nextState)
            # Save experience into memory replay buffer
            currState = environment.states[timeStep].toTensor()
            nextState = nextState.toTensor()
            agent.storeMemory(currState, action.index, reward, nextState, done)
            if timeStep % sp.learnStep != 0:
                status = environment.step()        
                continue
            # Draw a mini-batch of experiences from the memory buffer
            sample = agent.sampleMemory()
            batch = sample[1:] 
            # if there are enough memories to get a batch 
            if sample[0]:
                # split the batch according to its elements  
                states, actions, rewards, nextStates, dones = batch
                # get the optimal Q value
                estimate = agent.td_estimate(states, actions)
            # continue to next timestep 
            else:
                # Increment time step 
                status = environment.step()
                continue
            # If the next state is terminal
            if done:
                # Set the target value as the current reward
                target = T.tensor(rewards).to(agent.targetNetwork.device)
            # Set the target value as the current reward and the expected future reward with a discount factor 
            else:
                target = agent.td_target(rewards, nextStates, dones)
            ##Update online network 
            loss = agent.updateOnlineQ(estimate, target)
            if sp.writeOutput:
                print('Loss: ', loss)
            # Add loss value to the window
            lossMetric.addValue(loss)
            # Soft-update the target network 
            agent.softUpdate(sp.tau)
            # End episode if next state is terminal 
            if done:
                done = False
                break
            # Increment time step 
            status = environment.step()
        # Update epsilon value
        agent.decrementEpsilon()
        print('epsilon value: ', agent.epsilon)
        # Get the average score from all the timesteps 
        if saveMetrics:
            rewardsMetric.sum()
            lossMetric.average()
            rewardsMetric.reset()
            lossMetric.reset()
        # Create density maps every N episodes 
        if saveMetrics and (episode == 0 or episode%sp.nbEpisodes == 0):
            metrics.merge(environment.world, episode, False)
            # metrics.userScatter(environment.world, episode, 'processed')
            # metrics.userScatter(environment.world, episode, 'generated')
            # metrics.userScatter(environment.world, episode, 'notCompleted')
        # total time steps per episode
        if saveMetrics:
            stepsMetric.addValue(timeStep)
            completedMetric.sum()
            completedMetric.reset()
            # expired tasks per episode 
            expiredMetric.sum()
            expiredMetric.reset()
            # total tasks generated 
            generatedMetric.sum()
            generatedMetric.reset()
        # Remove the stored scores and uav positions
        metrics.resetWindows()
finally:
    print('-------- Termination --------')
    plot = True
    if plot:
        if saveMetrics:
            completedMetric.scatter(episode, True)
            expiredMetric.scatter(episode, True)
            generatedMetric.scatter(episode, True)
            stepsMetric.scatter(episode, False)
            rewardsMetric.scatter(episode, True)
            lossMetric.scatter(episode, True)
            offloadMetric.histogram()
    else:
        pass