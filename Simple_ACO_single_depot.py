''' ACO Signle Depot - Multi Agents '''
''' Importing Libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random

''' Departure Function'''
def departure(alpha, beta):

    Tij_times_visibility = np.zeros((cities_num, cities_num))              # To save new Tij*visibility for making a decision

    ''' Probability of moving for each agent'''
    for agent in range(agents_in_depot):
        '''visibility to beta and pheromone value to alpha'''
        for i_index in range(cities_num):
            for j_index in range(cities_num):
                Tij_times_visibility[i_index, j_index] = Tij[i_index, j_index] ** alpha * visibility[i_index, j_index] ** beta
        ''' Decide based on Tij in Visibility'''

        if np.sum(Tij_times_visibility[agents_path[agent][-1]][:]) != 0:  # Agent moves if there is a node unvisited
            agents_probability[agent][:] = Tij_times_visibility[agents_path[agent][-1]][:]/np.sum(Tij_times_visibility[agents_path[agent][-1]][:])
            agents_probability_cumsum[agent][:] = agents_probability[agent][:].cumsum()
            rand_num = random.uniform(0,1)
            move(agent,rand_num)
        else: print('All customers are visited.')

''' Move Function'''
def move(agent, rand_num):
    flag = True
    for i_index in range(cities_num):
        if rand_num <= agents_probability_cumsum[agent][i_index] and agents_probability[agent][i_index] != 0:  # Choose this city i_index
            for j_index in range(cities_num):               # This city is visited so visibility is turned into 0
                visibility[j_index][i_index] = 0
            agents_path[agent].append(i_index)              # Assigning agent to the selected city
            agents_tour_lenght[agent] += distance[agents_path[agent][-2]][agents_path[agent][-1]]
            flag = False
            break
    if flag == True:
        print('Error: No destination is chosen')

''' Update Pheromone Function'''
def update_pheromone(rho, Tij):
    Tij = np.dot(Tij,(1-rho))  # First change applicable on all routes
    for agent in range(agents_in_depot):
        origin = agents_path[agent][-2]; destin = agents_path[agent][-1]
        Tij[origin][destin] += 1/agents_tour_lenght[agent]
    return Tij

'''Main'''
if __name__ == '__main__':
    '''Initialization of ACO algorithm'''
    cities_x = [4, 2, 1, 4, -2, 1, 5]  # Cities X cordinator ==> np.array
    cities_y = [4, 5, 6, -2, 3, 4, 8]  # Cities Y cordinator ==> np.array
    cities_num = len(cities_y)  # Number of cities    ==> np.array

    # Status of cities being visited 0: no, 1: yes ==> np.array
    cities_visited = [np.linspace(1, cities_num, cities_num), np.zeros(cities_num)]

    depot_x = [4]  # Depots X Cordinator ==> np.array
    depot_y = [4]  # Depots Y cordinator ==> np.array

    # plt.scatter(cities_x,cities_y); plt.scatter(depot_x, depot_y, c='red'); plt.show()

    agents_in_depot = 3  # Number of agents in each depot ==> int
    agents_path = [[0], [0], [0]]  # Agents' chosen path
    agents_tour_lenght = [0, 0, 0]  # Agents' tour lenght

    agents_probability = np.zeros((agents_in_depot, cities_num))  # Temporary matrix to save probabilities
    agents_probability_cumsum = np.zeros(
        (agents_in_depot, cities_num))  # Temporary matriX to save cumulative probabilities for each agent

    Tij = np.ones((cities_num, cities_num))  # Pheromone value between cities ==> at first is 1
    distance = np.zeros((cities_num, cities_num))  # Distance between cities     ==> np.array
    visibility = np.zeros((cities_num, cities_num))  # Visibility between cities   ==> np.array

    # Creation of the first visibility matrix
    for origin in range(cities_num):
        for destin in range(cities_num):
            dist = math.dist((cities_x[origin], cities_y[origin]), (cities_x[destin], cities_y[destin]))
            distance[origin, destin] = dist
            if dist == 0:
                visibility[origin, destin] = 0
            else:
                visibility[origin, destin] = 1 / dist

    alpha = 1           # Pheromone importance
    beta = 1            # Degree of visibility
    rho = 0.2           # Evaporation rate
    iteration = 4       # Iteration Number
    cost_per_unit = 0.3   # Per distance

    for iter in range(iteration):                # Continue running for 500 times
        departure(alpha, beta)
        Tij = update_pheromone(rho, Tij)
        if iter == 0:                        # After one iteration, agents assigned to nodes and depot is empty
            for i_index in range(cities_num):
                visibility[i_index][0] = 0
                visibility[0][i_index] = 0

    ''' Calculation of traveling costs'''
    traveling_cost = sum(agents_tour_lenght)*cost_per_unit

    ''' Printing & Plotting Results'''
    print("Agents' path is: ", agents_path)
    print('Traveling Cost Is: %0.2f' % traveling_cost)

    colours = pd.read_csv('My colours.csv')
    colours = list(colours.columns)

    ''' Add the last destination which is the depot'''
    label_maker = np.ones(agents_in_depot)
    for agent in range(agents_in_depot):
        agents_path[agent].append(0)
        agent_x = []; agent_y =[]
        for node in agents_path[agent]:
            agent_x.append(cities_x[node])
            agent_y.append(cities_y[node])

            if label_maker[agent] == 1:
                plt.plot(agent_x, agent_y, c=colours[2*agent],label=f'Agent #{agent}')
                label_maker[agent] = 0
            else:
                plt.plot(agent_x, agent_y, c=colours[2 * agent])

    for city in range(cities_num):
        plt.text(cities_x[city], cities_y[city], f'{city}', horizontalalignment='center')
        plt.plot(cities_x[city], cities_y[city],'o', c='orange')
        
    plt.plot(depot_x, depot_y, 'bo')
    plt.xlabel('X axes')
    plt.ylabel('Y axes')
    plt.legend()
    plt.title('Paths Passed By Agents')
    plt.show()

    ''' Save distance file'''
    pd.DataFrame(distance).to_csv('distance.csv')
