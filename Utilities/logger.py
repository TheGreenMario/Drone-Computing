"""
Module that has the classes to take metrics of the network and environment.
listLog is a class to take metrics from any values that can be stored in a list, and perform the subsequent plotting.

- Save reward per episode
- Save loss per episode 
- Save user's positions
- Save UAV positions 
- Save the number of tasks processed per user per episode
- Save number of total time steps per episode
- Save total tasks processed per episode  
- Save task offloading percentages 


These metrics are then used to draw various graphs, including scatter plots, density maps, heat maps and line graphs.

----
"""
import matplotlib.pyplot as plt
import os
import numpy as np
import plotly.graph_objects as go
import Utilities.utils as utils
import simParam as sp
import plotly.express as px

class Logger():
    """ Class to represent a logging tool

    :param savePath: Folder path to save recorded metrics
    :type savePath: string
    :param rewards: List to store the values of the rewards 
    :type rewards: list[float]
    :param uavPositions: Array to store the positions of the uav 
    :type uavPositions: ndarray
    """

    def __init__(self, savePath):
        # path to save metrics 
        self.savePath = savePath
        self.epDir()
        # list to store reward values
        self.rewards = []
        self.scoreWindow = []
        # list to store loss values 
        self.loss = []
        self.lossWindow = []
        # list to store UAV positions
        self.uavPositions = np.zeros((1,3))

    def epDir(self):
        """Create directory to store the episodic metrics 
        """
        # setup folder for metrics taken per episode
        self.heatPath = self.savePath + '/episodic/'
        # create folder if it does not exist 
        if not os.path.isdir(self.heatPath):
            os.mkdir(self.heatPath)

    def addPosition(self, position):
        """Add a UAV position to the uavPositions array 

        :param position: UAV position to be saved 
        :type position: ndarray
        """
        self.uavPositions = np.append(self.uavPositions, [position], axis=0)

    def addScore(self, reward):
        """Add a new reward value to the reward list

        :param reward: Reward value to be added
        :type reward: float
        """
        length = len(self.scoreWindow)
        self.scoreWindow.append(reward)
        
    def averageList(self, list):
        """Calculate the average reward of the current score list

        :param list: List whose values should be averaged
        :type list: list[float]

        :return: Average of the score window
        :rtype: float
        """
        if len(list) > 0:
            average = sum(list)/len(list)
        else:
            average = 0
        return average
    
    def resetWindows(self):
        """Clear the score window list for the next episode
        """
        self.scoreWindow = []
        self.lossWindow = []
        self.uavPositions = np.zeros((1,3))

    def listToFile(self, fileName, list, episode = -1):
        """Write the provided list to a file on the metric's savePath folder

        :param fileName: Name of the file to be writen 
        :type fileName: string
        :param list: Name of the list whose contents will be writen to the file
        :type list: list[]
        :param episode: Current episode number of the simulation, defaults to -1 (episode number should not be used in naming the file)
        :type episode: int
        """
        # path to save the file 
        if episode != -1:
            fileName = self.savePath + fileName + '_' +  str(episode) + '.txt' 
        else:
            fileName = self.savePath + fileName
        # open the file 
        with open(fileName, 'w') as file:
            # write all values on the list to the file 
            for value in list:
                file.write(str(value)+ '\n')


    def plotFromFile(self, dataName, fileName):
        """Create a data plot from a file 

        :param dataName: Name that should be given to the data (in the plot and file)
        :type dataName: string
        :param fileName: Name of the file whose data will be plotted
        :type fileName: string 

        :return: 1: file exists | 0: provided fileName does not exist
        :rtype: int
        :raises Exception: File not found
        """
        #  path to save the file 
        fileName = self.savePath + fileName

        # check if file exists
        if os.path.isfile(fileName):
            data = []
            # open file for read-only operation 
            with open(fileName, 'r') as file:
                for line in file:
                    data.append(float(line.strip()))

            # plot the data
            plt.plot(data)

            # Plot display settings
            plt.title(dataName + ' Plot')
            plt.xlabel('Episodes')
            plt.ylabel(dataName)
            # Display grid
            plt.grid()

            # Save a copy of the plot image 
            savePathStore = self.savePath + dataName + '.png'
            plt.savefig(savePathStore)
            # Close figure 
            plt.close()
            return 1
        else:
            # if file doesnt exist 
            raise Exception('File not found: ', fileName)
        
    def uavDensityMap(self): 
        """ Plot a density map with the saved UAV positions
        """ 
        # get (x,y) values from the positions array 
        x = self.uavPositions[:,0]
        y = self.uavPositions[:,1]
        fig = go.Figure(go.Histogram2d(
                x=x,
                y=y,
            ))
        fig.show()
        filePath = self.savePath + 'densityMap' + '.html'
        fig.write_html(filePath)

    def tasksHeatMap(self, world, episode, show=False): 
        """Plot a heat map with the number of tasks processed per user and save it locally

        :param world: World with the list of corresponding users
        :type world: World 
        :param episode: Current episode of the algorithm
        :type episode: int
        :param show: True if figure should be displayed on the browser, defaults to False
        :type show: boolean
        """
        # Initialize matrix
        matrix = np.zeros((sp.maxBound, sp.maxBound))
        # Loop to add entries 
        for row in range(sp.maxBound):
            for column in range(sp.maxBound):
                id = world.generateID(row, column)
                user = world.userFromID(id)
                if user: 
                    matrix[column][row] = user.processed
                else:
                    matrix[column][row] = -1
        # create array of numbers 1 to n
        n = sp.maxBound
        # Define the axis of the plot 
        x = np.arange(0, n, 1)
        y = np.arange(0, n, 1)
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
          x = x,
          y = y,
          z = matrix,
          type = 'heatmap',
          colorscale = 'Picnic'))
        fig.update_layout(
            width = 700, height = 700,
            autosize = False,
            hovermode='closest')
        # show the final result 
        if show:
            fig.show()
        # Save html file on heatmap output folder
        filePath = self.heatPath + 'taskHeat_' + str(episode) + '.html'
        fig.write_html(filePath)

    def processedScatter(self, world, episode, show=False):
        """Create a scatter plot of the number of tasks processed per user and save it locally

        :param world: World with the list of corresponding users
        :type world: World 
        :param episode: Current episode of the algorithm
        :type episode: int
        :param show: True if figure should be displayed on the browser, defaults to False
        :type show: boolean
        """
        # Get all users
        users = world.users
        # Initialize loop variables 
        positions = np.zeros((len(users), 3))
        count = 0
        # Get the position and tasks processed from each user
        for user in users:
            item = [user.getX(), user.getY(), user.processed]
            positions[count] = item
            count = count + 1
        # Draw the scatter plot 
        fig = px.scatter(x=positions[:,0], y=positions[:,1], color=positions[:,2],
                        title="Tasks processed per user",
                        width = 700, height = 700,
                        labels={'x':'User Position X', 'y':'User Position Y', 'color': 'Tasks processed'},
                        color_continuous_scale=['blue', 'red'])
        # If plot should be displayed in the browser
        if show:
            fig.show()
        # Save html file on heatmap output folder
        filePath = self.heatPath + 'processedScatter_' + str(episode) + '.html'
        fig.write_html(filePath)

    def userScatter(self, world, episode, variable, show=False):
        """Create a scatter plot of a given user variable

        :param world: World with the list of corresponding users
        :type world: World 
        :param episode: Current episode of the algorithm
        :type episode: int
        :param show: True if figure should be displayed on the browser, defaults to False
        :type show: boolean
        """
        # Get all users
        users = world.users
        # Initialize loop variables 
        positions = np.zeros((len(users), 3))
        count = 0
        # Get the position and tasks processed from each user
        for user in users:
            attribute = getattr(user, variable)
            item = [user.getX(), user.getY(), attribute]
            positions[count] = item
            count = count + 1
        # Draw the scatter plot 
        fig = px.scatter(x=positions[:,0], y=positions[:,1], color=positions[:,2],
                        title=f"Tasks {variable} per user",
                        width = 700, height = 700,
                        labels={'x':'User Position X', 'y':'User Position Y', 'color': f'Tasks {variable}'},
                        color_continuous_scale=['blue', 'red'])
        # If plot should be displayed in the browser
        if show:
            fig.show()
        # Save html file on heatmap output folder
        filePath = f'{self.heatPath} {variable}Scatter_{episode}.html'
        fig.write_html(filePath)



    def scatterDifferences(self, world, episode):
        """Plot a scatter plot of the different elements between two lists

        """
        # Get all users
        users = world.users
        # Initialize loop variables 
        positions = np.zeros((len(users), 3))
        count = 0
        # Get the position and tasks processed from each user
        for user in users:
            attribute = getattr(user, 'notCompleted')
            item = [user.getX(), user.getY(), attribute]
            positions[count] = item
            count = count + 1
        # Draw the scatter plot 
        fig = px.scatter(x=positions[:,0], y=positions[:,1], color=positions[:,2],
                        title=f"Tasks not completed per user",
                        width = 700, height = 700,
                        labels={'x':'User Position X', 'y':'User Position Y', 'color': f'Tasks not completed'},
                        color_continuous_scale=['blue', 'red'])
        # Save html file on heatmap output folder
        filePath = f'{self.heatPath} notCompletedScatter_{episode}.html'
        fig.write_html(filePath)


    def scatterPlot(self, file): 
        """ Create a scatter plot from the provided file and display it (plot is not saved)

        :param file: Name of the file whose data will be used to create the scatter plot
        :type file: string
        """
        # path to users file 
        fileName = self.savePath + file
        # check if file exists
        if os.path.isfile(fileName):
            data = []
            # open file for read-only operation 
            with open(fileName, 'r') as f:
                for line in f:
                    data.append(line)

        positions = np.zeros((1,3))
        # get positions
        for index in range(len(data)):
            item = utils.getIntFromString(data[index])
            positions = np.append(positions, [item], axis=0)
        # Delete position from array creation 
        positions = np.delete(positions, 0, 0)
        # draw the scatter plot 
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=positions[:,0],
            y=positions[:,1],
            mode='markers',
            showlegend=False,
            marker=dict(
                symbol='x',
                opacity=1,
                color='black',
                size=12,
                line=dict(width=1),
            )
        ))
        fig.show()

    def merge(self, world, episode, show=True): 
        """ Function to overlay the user position scatter plot with the UAV position density map and display it on the web browser
        (and save it locally)

        :param world: World instance with the UAV positions to be users to be plotted
        :type world: World
        :param show: True if the graph should be displayed, defaults to True
        :type show: boolean
        """
        # add edge position to the density map
        match sp.mapType:
            case 'rectangle':
                self.addPosition(([sp.x2, sp.y2 ,0]))
            case 'square':
                self.addPosition(([sp.maxBound, sp.maxBound ,0]))
        # get (x,y) values from the uav positions array 
        xUav = self.uavPositions[:,0]
        yUav = self.uavPositions[:,1]
        # Get all users
        users = world.users
        # Initialize loop variables 
        positions = np.zeros((len(users), 3))
        count = 0
        # Get the position and tasks processed from each user
        for user in users:
            item = [user.getX(), user.getY(), user.processed]
            positions[count] = item
            count = count + 1
        fig = go.Figure()
        # draw the scatter plot 
        fig.add_trace(go.Scatter(
            x=positions[:,0],
            y=positions[:,1],
            mode='markers',
            showlegend=False,
            marker=dict(
                symbol='x',
                opacity=1,
                color='black',
                size=12,
                line=dict(width=1),
            )
        ))
        # add the density map
        fig.add_trace(go.Histogram2d(
            x=xUav,
            y=yUav,
            colorscale= 'YlGnBu',
            nbinsx= 100,#math.ceil((sp.maxBound + 1)/(sp.uavMaxSpeed)),
            nbinsy= 100,#math.ceil((sp.maxBound + 1)/(sp.uavMaxSpeed)),
            zauto=False,
        ))
        fig.update_layout(
            xaxis=dict( ticks='', showgrid=True, zeroline=False, nticks=20 ),
            yaxis=dict( ticks='', showgrid=True, zeroline=False, nticks=20 ),
            autosize=False,
            height=700,
            width=700,
            hovermode='closest',
        )
        # show the final result 
        if show:
            fig.show()
        # Save html file on heatmap output folder
        filePath = self.heatPath + 'heatMap_' + str(episode) + '.html'
        fig.write_html(filePath)

class listLog(): 
    """ Class to represent a logging tool using metrics stored in lists

    :param savePath: Folder path to save recorded metrics
    :type savePath: string
    :param values: List of values saved   
    :type values: list[any]
    :param window: List to store sum/average of data from the values list
    :type window: list[any]
    :param dataName: Name to identify the plot of the data
    :type dataName: string 
    """
    
    def __init__(self, savePath, dataName):
        self.savePath = savePath
        self.values = []
        self.window = []
        self.dataName = dataName

    def addValue(self, value):
        """
        Add value to the values list

        :param value: Value to be appended to the list
        :type value: any
        """
        self.values.append(value)

    def reset(self):
        """Reset the values list to be empty 
        """
        self.values = []

    def average(self):
        """Average the values in the list and append them to the window list 
        """
        if len(self.values) > 0:
            self.window.append(sum(self.values)/len(self.values))
        else:
            self.window.append(0)
            
    def sum(self):
        """Sum the values in the list and append them to the window list 
        """
        self.window.append(sum(self.values))

    def scatter(self, episodes, window = False, show = False):
        """
        Create a scatter plot with the data stored in the values list 

        :param episodes: Number of episodes processed until completion
        :type episodes: int
        :param window: True if the window values should be used, defaults to False (using the values list)
        :type window: boolean
        :param show: True if plot should be automatically displayed, defaults to False
        :type show: boolean
        """
        # x axis represents the number of episodes 
        x = np.arange(start=0, stop=episodes, step=1)
        # y axis represents the data values to be plotted 
        if window:
            y = self.window
        else:
            y = self.values
        # create the scatter plot 
        fig = go.Figure(data=go.Scatter(x=x, y=y, 
                                        mode='lines+markers',
                                        name=self.dataName,
        ))
        # Create the title of the plot 
        title = self.dataName + ' per Episode'
        # Update the labels 
        fig.update_layout(title=title,
                          xaxis_title = 'Episodes',
                          yaxis_title = self.dataName)
        # Choose wether the plot should be displayed on the browser
        if show: 
            fig.show()
        # Save html file on heatmap output folder
        filePath = self.savePath + self.dataName + '.html'
        fig.write_html(filePath)

    def histogram(self, show = False):
        """
        Create an histogram based on the values list 

        :param show: True if the plot should be displayed on the browser, defaults to False
        :type show: boolean
        """
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x = self.values,
            name= self.dataName,
        ))
        fig.update_layout(
            title_text=self.dataName, # title of plot
            xaxis_title_text='Value', # xaxis label
            yaxis_title_text='Count', # yaxis label
            bargap=0.2, # gap between bars of adjacent location coordinates
        )
        # Choose wether the plot should be displayed on the browser
        if show:
            fig.show()
        # Save html file on heatmap output folder
        filePath = self.savePath + self.dataName + '.html'
        fig.write_html(filePath)
        
    def scatterXY(self, x, y, show = False):
        """
        Create a scatter plot with the given x and y values for the data

        :param x: Values to be used in the x axis
        :type x: List[]
        :param y: Values to be used on the y axis
        :type y: List[]
        :param show: True if plot should be automatically displayed, defaults to False
        :type show: boolean
        """
        # create the scatter plot 
        fig = go.Figure(data=go.Scatter(x=x, y=y, 
                                    mode='lines+markers',
                                    name=self.dataName,
        ))
        # Create the title of the plot 
        title = self.dataName
        # Update the labels 
        fig.update_layout(title=title,
                        xaxis_title = 'Distance (m)',
                        yaxis_title = self.dataName)
        # Choose wether the plot should be displayed on the browser
        if show: 
            fig.show()
        # Save html file on heatmap output folder
        filePath = self.savePath + self.dataName + '.html'
        fig.write_html(filePath)