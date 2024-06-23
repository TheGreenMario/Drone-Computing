"""
Module with the functions to manage and update the task tag of the environment

"""

from .MDP import Environment, UAV, Action, User, Task

def actionTimeline(environment: Environment, uav: UAV, action: Action, user: User):
    """Update the task tag according to the state of the task being processed (task in progress, new task or no task)

    :param environment: Environment whose Task Tag will be updated 
    :type environment: Environment 
    :param uav: UAV used to determine the new action
    :type uav: UAV
    :param action: Action selected based on the current state
    :type action: Action
    :param user: User whose task is being processed 
    :type user: User
    """
    init = False
    # Check if a task is currently being processed 
    if environment.tag.task != None:
        # Calculate d value and steps to complete for a new task 
        if environment.tag.stepstoComplete == 0:
            environment.newAction(uav, action.taskOffload, action.positionDelta, user)
            environment.tag.getDuration()
            init = True
        environment.tag.incrementSteps()
        # Check if a task was not completed after the chosen action 
        if not environment.tag.completed:
            if environment.tag.task.d != None:
                if not init:
                    # If the task was not completed, maintain the same 'd' value for the next state
                    environment.newAction(uav, environment.tag.task.d, action.positionDelta, user)
                    environment.tag.getDuration()
                else:
                    init = False
            else:
                # If it is a new task (no 'd' value associated yet)
                environment.newAction(uav, action.taskOffload, action.positionDelta, user)
                environment.tag.getDuration()
        else:
            # If the task was completed, keep the d value for the rest of the timestep calculations
            environment.newAction(uav, environment.tag.task.d, action.positionDelta, user)
    else:
        # Choose action considering there is no task being processed 
        environment.newAction(uav, action.taskOffload, action.positionDelta, user)

def preprocessTimeline(environment: Environment, uav: UAV):
    """Update the task tag and the selected user depending on the next task to be processed (the same, a new one or none)

    :param environment: Environment whose tag will be updated 
    :type environment: Environment 

    :return: Selected user or None if there are no tasks in the queue 
    :rtype: User  or None 
    """
    # If the task was completed or there is no task 
    if environment.tag.task == None:
        # Get next task from queue 
        nextTask: Task = uav.getTopTask()
        if nextTask != None:
            # Get user associated with task and add it to the tag 
            user = environment.world.userFromID(nextTask.userID)
            environment.tag.updateTask(nextTask)
        else:
            # If there is no task, reset tag and user 
            user = None
            environment.tag.reset()
    else:
        # Keep processing the same task 
        nextTask = environment.tag.task
        user = environment.world.userFromID(nextTask.userID)

    return user