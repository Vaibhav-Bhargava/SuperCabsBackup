# Import routines

import numpy as np
import math
import random
from itertools import product

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

Time_matrix = np.load("TM.npy")


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        # Actions are supposed to be locations, from 0 to 4 plus (0,0) for idling
        l1 = l2 = list(np.arange(0,m))
        # get all combinations of the locations
        l3= product(l1,l2)
        # Convert the iterator to list
        l3 = [list(i) for i in l3]
        # remove the combinations for same pickup and drop locations
        for i in l3[1:]:
            if i[0]==i[1]:
                l3.remove(i)
        
        # The action space would also have state (0,0) for not accepting any requests
#         l3.append((0,0))
        self.action_space = l3
        self.state_space = [[i,j,k] for i in range(m) for j in range (t) for k in range(d)]
        # let's start the episode with a random state
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        #So, we are creating a one-hot encoded vector of size 5+24+7
        # Let's create an empty state encoded vector of size m + t + d
        state_encod = [0] * (m + t + d)
        # First part is the position. Set the element at position state[0] as 1
        state_encod[state[0]]=1
        # the time part starts after position. So there would be an offset of m
        # Now, let's set the time vector
        state_encod[m + state[1]] = 1
        # the offset for day vector would be m + d
        state_encod[m + t + state[2]] = 1

        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(15)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)
        # Capping the number of requests at 15
        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m+1), requests) # (0,0) is not considered as customer request
        # But, the driver can always refuse to take any request at any location. So, we will add index of [0,0], which is 0
        possible_actions_index = [0]+(possible_actions_index)
        actions = [self.action_space[i] for i in possible_actions_index]

        return possible_actions_index,actions
    
    
    def new_time_and_day(self,time,day,time_elapsed):
        """
        Checks, if the day changes after the ride gets over or the driver idles
        Also, change the time back to 24 hr format if time goes over 23 hrs
        """
        new_time = 0
        new_day = 0
        if (time + time_elapsed) < 24:
            new_time = time + time_elapsed
            new_day = day
        else:
            # Convert time to 24 hour format
            new_time = (time + time_elapsed) % 24
            # Check, how many days passed by
            days = (time + time_elapsed)//24
            new_day = day + days
            # But, there are only seven days in a week. So, if new_day exceeds 6, it has to be brought back to 0-6 day format
            new_day = new_day % 7
        
        return int(new_time), int(new_day)



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward
        There can be three cases, which would affect the reward:
            i) Driver is at the same location as the pickup location for next ride
            ii) Driver has to go to the pick location from his location
            iii) Driver doesn't take the new request and idles away for next 1 hour
        So, the total time would be sum of idle time, transit time to pickup and the ride time
        The reward is a function of time. 𝐶𝑓 if the battery cost per hour and 𝑅𝑘 is the revenue earned
        As per the equation in the problem statement:
        𝑅(𝑠 = 𝑋𝑖𝑇𝑗𝐷𝑘) = 𝑅𝑘 ∗ (𝑇𝑖𝑚𝑒(𝑝, 𝑞)) − 𝐶𝑓 ∗ (𝑇𝑖𝑚𝑒(𝑝, 𝑞) + 𝑇𝑖𝑚𝑒(𝑖, 𝑝)) 𝑎 = (𝑝, 𝑞)
                                         - 𝐶𝑓                          𝑎 = (0,0)
        We would start with initialising all three times to 0
        """
        idle_time = 0
        pickup_transit_time = 0
        ride_time = 0
        initial_location = state[0]
        pickup_location = action[0]
        drop_location = action[1]
        start_time = state[1]
        end_time = None
        start_day = state[2]
        end_day = None
        
        # Case 1: Driver is at the same location as the pickup and s/he takes a ride
        if(initial_location==pickup_location and action!=[0,0]):
            ride_time = Time_matrix[initial_location][drop_location][start_time][start_day]
        # Case 2: Driver is not at the same location as pickup and s/he takes a ride
        if(initial_location!=pickup_location and action!=[0,0]):
            pickup_transit_time = Time_matrix[initial_location][pickup_location][start_time][start_day]
            # Checking, in case the day changes
            pickup_time, pickup_day = self.new_time_and_day(start_time,start_day,pickup_transit_time)
            ride_time = Time_matrix[pickup_location][drop_location][pickup_time][pickup_day]
        # Case 3: The driver decides not to take the ride
        if (action==[0,0]):
            idle_time = 1
        """
        Now, let's calculate the reward
        """
        # the driver is idling i.e. doesn't take any ride requests
        if (action==[0,0]):
            reward = -(C)
        # Else, if the driver decides to take a ride
        else:
            reward = R * ride_time - C * (pickup_transit_time + ride_time)
        
        
        return int(reward)




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state
        There can be three cases, which would affect the next state:
            i) Driver is at the same location as the pickup location for next ride
            ii) Driver has to go to the pick location from his location
            iii) Driver doesn't take the new request and idles away for next 1 hour
        So, the total time would be sum of idle time, transit time to pickup and the ride time
        We would start with initialising all three times to 0
        """
        idle_time = 0
        pickup_transit_time = 0
        ride_time = 0
        initial_location = state[0]
        pickup_location = action[0]
        drop_location = action[1]
        start_time = state[1]
        end_time = None
        start_day = state[2]
        end_day = None
        
        # Case 1: Driver is at the same location as the pickup and s/he takes a ride
        if(initial_location==pickup_location and action!=[0,0]):
            ride_time = Time_matrix[initial_location][drop_location][start_time][start_day]
        # Case 2: Driver is not at the same location as pickup and s/he takes a ride
        if(initial_location!=pickup_location and action!=[0,0]):
            pickup_transit_time = Time_matrix[initial_location][pickup_location][start_time][start_day]
            # Checking, in case the day changes
            pickup_time, pickup_day = self.new_time_and_day(start_time,start_day,pickup_transit_time)
            ride_time = Time_matrix[pickup_location][drop_location][pickup_time][pickup_day]
        # Case 3: The driver decides not to take the ride
        if (action==[0,0]):
            idle_time = 1
            # if the driver doesn't take the ride, the drop location is same as the initial location
            drop_location = initial_location
        
        """ 
        The total time is the sum of idle, pickup transit and ride time
        if the driver doesn't take a ride, idle time would be 1 hour and pickup transit and ride time would be zero
        If the driver does decide to take a ride, idle time would be zero and other times would have values, depending on 
        pickup location
        """
        total_ride_time = idle_time + pickup_transit_time + ride_time
        
        end_time, end_day = self.new_time_and_day(start_time, start_day, total_ride_time)
        
        next_state = [drop_location, end_time, end_day]
        
        return next_state, total_ride_time
    
    def take_step(self, state, action, Time_matrix):
        next_state, total_ride_time = self.next_state_func(state, action, Time_matrix)
        reward = self.reward_func(state, action, Time_matrix)
        return next_state, reward, total_ride_time


    def reset(self):
        return self.action_space, self.state_space, self.state_init
