# -*- coding: utf-8 -*-
"""
Created on Sun Jul 01 15:29:15 2018
@author: pp34747
"""
import random
import numpy as np
import math
import candidates
import graphs
import matplotlib.pyplot as plt
import os
import psutil
import time
from itertools import chain, izip
import graph_tool.all as gt
import dynamic_connectivity as dc
from scipy.stats.mstats import gmean

class Optimize():
    def __init__(self, cost=None, nIterations=200):
        self.T_start = 1.0
        self.T_min = 0.000000001
        self.cost = cost
        self.nIterations = nIterations
        
    def run(self, candInfo, conflictList, mergeList, adjacencyList, dummyConflictsList, curvature_offset_dict, ref_value_offset_dict):
        
        pid = os.getpid()
        py = psutil.Process(pid)            
            
        print "Resolving root hairs ..."
        nCandidates = len(conflictList)

        if nCandidates > 1: # If more than one candidate we need to find optimal set of roothairs
                        
            # Set initial solution and parameters
            print " - Resolving..."
            
            # Determine number of iterations
            maxLevels = 5*self.nIterations     
            
            # Initialize state object
            state = State(  candInfo=candInfo,
                            mergeList=mergeList,
                            conflictList=conflictList,
                            adjacencyList=adjacencyList,
                            dummyConflictsList=dummyConflictsList,
                            curvature_offset_dict=curvature_offset_dict,
                            ref_value_offset_dict=ref_value_offset_dict)


            # Create random state
            shuffleState(state,nCandidates)

            # Get values for cooling schedule by simulating random states
            finalProb = 0.01/nCandidates                                # 1% chance of upward move being accepted at last temperature level
            initialProb = 0.95                                          # 95% chance of upward move being accepted at initial temperature level
            csMaker = CoolingScheduleMaker(state, costFunction=self.cost, initialProb=initialProb, finalProb=finalProb) # object to make cooling schedule
            csMaker.simulate(int(0.2*self.nIterations*nCandidates))     # 20% of iterations in actual optimization
            normArray = csMaker.normalization()                         # calculates avergae values of sub costs for normalization
            self.cost.setNormValues(normArray)                          # set normalization value in cost function
            csMaker.recalculateCost()                                   # calculate costs with normalization
            csMaker.calculateUpwardCosts()                              # calculate average upword cost
            initialTemp = csMaker.getInitialTemp()                      # calculate initial temperature
            alpha = csMaker.getAlpha(initialTemp, self.nIterations)     # calculate alpha
            finalTemp = csMaker.getFinalTemp(initialTemp,alpha, self.nIterations)   # calculate final temperature
            averageCost = csMaker.initialCost()                         # calcuulate initial cost from average of all costs

            print("norm curvature: "+str(1./normArray[0])+", norm length: "+str(1./normArray[1])+". norm distance: "+str(1./normArray[2]))
            print "averageCost:"+str(averageCost), "averageDeltaCost: "+str(csMaker.averageDeltaCost()), ", initialTemp: ", str(initialTemp), ", finalTemp: ", str(finalTemp), ", alpha: ", str(alpha)


            # Initialize Simulated Annealing object
            sa = SimulatedAnnealing(initialState=state, initalTemp=initialTemp, finalTemp=finalTemp, averageCost=averageCost, alpha=alpha, maxLevels=maxLevels, costFunction=self.cost)


            memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
            print ' - memory use: '+ str(round(memoryUse,4))
            
            start_time = time.time()
            # Run optimization
            best_sol, best_cost, ratio_complete, bestMetricsNorm, n_iterations = sa.anneal()
            memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
            print ' - memory use: '+ str(round(memoryUse,4))
            #summary[i_run+1] = {'n':n_candidates, 'T':T_arr, 'cost':cost_arr, 'best':best_arr}               
            
            # Final solution  
            sol_out = np.where(best_sol)[0]
                        
            elapsed_time = time.time() - start_time
            print 'Time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        
        elif nCandidates == 1: # If only one candidate -> trivial solution
            sol_out = [0]
        else: # If no candidates -> no solution
            sol_out = []
        
        memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
        print ' - memory use: '+ str(round(memoryUse,4))
        
        # Create roothair connected components from overall best solution
        components = dc.ConnectedComponents(mergeList)
        for v in sol_out:
            components.addVertex(v)
        
        # Construct paths of connected components
        roothair_paths = []
        for cc in components.components.values():
            g = graphs.Candidate_Graph([])
            for candidate_id in cc:
                g.merge(graphs.Candidate_Graph(candInfo.paths[candidate_id]))
            roothair_paths.append(g.get_path())
        
        memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
        print ' - memory use: '+ str(round(memoryUse,4))
        
        sa_parameters = {'SA_finalProb':finalProb,
                         'SA_initialProb':initialProb,
                         'SA_normCurvature': normArray[0],
                         'SA_normLength': normArray[1],
                         'SA_normDistance': normArray[2],
                         'SA_initialTemp': initialTemp,
                         'SA_alpha': alpha,
                         'SA_finalTemp': finalTemp,
                         'SA_averageCost': averageCost,
                         'SA_nIterations': n_iterations}

        solution_summary = {'SA_bestCost':best_cost,
                         'SA_bestCurvatureCost': bestMetricsNorm[0],
                         'SA_bestLengthCost': bestMetricsNorm[1],
                         'SA_bestDistanceCost': bestMetricsNorm[2],
                         'SA_ratioComplete': ratio_complete}

        return roothair_paths, solution_summary, sa_parameters

def shuffleState(state, n=100):

    for _ in range(n):

        # Get position (component id) to be changed
        position = random.randint(0,n-1)

        # Change current solution
        isvalid = state.neighbor(position)

        # If is invalid reverse and skip
        if not isvalid:
            state.reverseChanges()
            continue

class SimulatedAnnealing:
    def __init__(self, initialState, initalTemp, finalTemp, averageCost, alpha, maxLevels, costFunction):
                
        self.state = initialState
        self.currTemp = initalTemp
        self.finalTemp = finalTemp
        self.averageCost = averageCost
        self.alpha = alpha
        self.maxLevels = maxLevels
        self.costFunction = costFunction

        self.iterationsPerTemp = len(self.state.candInfo.paths) # Number of iterations per temperature level
        self.R_max = 2 * self.iterationsPerTemp
        self.R = 0

    def anneal(self):

        # Initialize cost
        metrics = self.costFunction.calculateMetrics(self.state)
        metricsNorm = self.costFunction.normalizeMetrics(metrics)
        cost = self.costFunction.calculateCost(metricsNorm)             # Initial best cost is initial cost

        # Save first solution as best solution
        best_sol = np.array(self.state.binaryList)                  # Initial best solution is initial solution

        new_cost = cost
        best_cost = cost
        best_metrics = metrics
        newMetricsNorm = metricsNorm
        bestMetricsNorm = metricsNorm
        
        # Uncomment for plotting images:
        #solutions_arr = [np.where(self.state.binaryList)[0]]    # List with all states
        #cost_arr = [cost]                                       # List with all costs
        #metrics_arr = [bestMetricsNorm]                         # List with all metrics

        

        n_iterations = 0 

        print " - " + str([n_iterations, "{0:.2E}".format(self.currTemp), round(cost,5), round(best_cost,5)]) \
                    + str([round(c,3) for c in metricsNorm]) + str(round(float(self.R)/self.R_max,3))

        while (self.currTemp > self.finalTemp or self.R < self.R_max) and n_iterations < self.maxLevels: # T must be less than finalTemp and R must be larger than R_max to stop

            for _ in range(self.iterationsPerTemp):

                # Get position (component id) to be changed
                position = random.randint(0,len(self.state.binaryList)-1)

                # Change current solution
                isvalid = self.state.neighbor(position)

                # If is invalid reverse and skip
                if not isvalid:
                    self.state.reverseChanges()
                    continue
                
                # Calculate cost for current state
                metrics = self.costFunction.calculateMetrics(self.state)
                newMetricsNorm = self.costFunction.normalizeMetrics(metrics)
                new_cost = self.costFunction.calculateCost(newMetricsNorm)
                
                

                # Acceptance probability
                ap = self.probability(self.averageCost, cost, new_cost, self.currTemp)
            
                # If acceptance probability is larger than random value between 0. and 1.
                if ap > random.random():
                    self.R = 0                      # Reset number of rejected moves
                    cost = new_cost                 # Cost is updated

                    if new_cost < best_cost:            # New cost is better than overall best cost
                        best_sol = np.array(self.state.binaryList)      # Update best solution
                        best_cost = new_cost                            # Update best cost
                        best_metrics = metrics
                        bestMetricsNorm = newMetricsNorm                # Update best metrics
                    
                    # Uncomment for plotting all accepted states:
                    #solutions_arr.append(np.where(self.state.binaryList)[0])
                    #cost_arr.append(new_cost)
                    #metrics_arr.append(newMetricsNorm)

                else:
                    self.state.reverseChanges()
                    self.R += 1                     # Increase number of consecutive rejeced moves

            # Reduce temperature
            self.currTemp = self.currTemp*self.alpha
                                
            # Increase number of iterations
            n_iterations += 1

            print " - " + str([n_iterations, "{0:.2E}".format(self.currTemp), round(cost,5), round(best_cost,5)]) \
                        + str([round(c,3) for c in bestMetricsNorm]) + str(round(float(self.R)/self.R_max,3))

        ratio_complete = 1.-best_metrics[1]
        # Uncomment for plotting images:
        #return best_sol, best_cost, ratio_complete, bestMetricsNorm, solutions_arr, cost_arr, metrics_arr

        return best_sol, best_cost, ratio_complete, bestMetricsNorm, n_iterations
        


    def probability(self,average_cost,prev_score,next_score,temperature):
        if next_score < prev_score:
            return 1.0
        else:
            return math.exp((prev_score-next_score)/average_cost/temperature)        
        



class State:
    def __init__(self, candInfo, mergeList, conflictList, adjacencyList, dummyConflictsList, curvature_offset_dict, ref_value_offset_dict):

        self.candInfo = candInfo
        self.binaryList = np.zeros(len(candInfo.paths),dtype=int)  #sol is a binary 1D numpy array (e.g. sol = array([0,1,0,1,1]))
        self.components = dc.ConnectedComponents(mergeList)
        self.conflictList = conflictList
        self.adjacencyList = adjacencyList
        self.dummyConflictsList = dummyConflictsList
        self.curvature_offset_dict = curvature_offset_dict
        self.ref_value_offset_dict = ref_value_offset_dict

        # Create a graph from path of each candidate
        self.candidate_graphs = []
        for p in self.candInfo.paths:
            self.candidate_graphs.append(graphs.Candidate_Graph(p))

        # Initialize individual items of cost
        # Add all dummies, because there are no candidate root hairs yet
        self.cost_items = CostItems(sum_length_dummy=sum(self.candInfo.dummy_lengths),
                                    sum_strain_dummy=sum(self.candInfo.dummy_strain),
                                    sum_length_all=sum(self.candInfo.dummy_lengths) )

        # No conflicts yet for dummies
        # Dummies can have more conflicting candidates in solution e.g. at intersection
        n_dummies = max([max(rh) for rh in dummyConflictsList if len(rh)>0])+1
        self.n_dummy_conflicts = np.zeros(n_dummies, dtype=int)

        # Items to track recent changes to candidates and connected components
        self.addedTips = []
        self.removedTips = []
        self.removedCandidates = []
        self.addedCandidates = []
        self.removedComponents = []
        self.addedComponents = []
        self.addedDummies = []
        self.removedDummies = []

        # Track difference in cost items
        self.cost_items_difference = CostItemDifference()



    def neighbor(self, position):
        
        # Reset previous tracked changes 
        self.addedTips = []
        self.removedTips = []
        self.removedCandidates = []
        self.addedCandidates = []
        self.removedComponents = []
        self.addedComponents = []
        self.addedDummies = []
        self.removedDummies = []
        self.cost_items_difference = CostItemDifference()

        # Get neighbor of current solution
        self.addedCandidates, self.removedCandidates = self.getChangesFromBinaryList(position)
        self.updateBinaryList(self.addedCandidates, self.removedCandidates)

        # Remove vertices from components graph
        for c in self.removedCandidates:
            comp_add, comp_remove = self.components.removeVertex(c)
            for value in comp_remove.values():
                self.removedComponents.append(value)
            for value in comp_add.values():
                self.addedComponents.append(value)
        
        # Add new vertices to components graph
        for c in self.addedCandidates:
            comp_add, comp_remove = self.components.addVertex(c)
            for value in comp_remove.values():
                self.removedComponents.append(value)
            for value in comp_add.values():
                self.addedComponents.append(value)
        
        # Fill with dummies
        self.addedDummies, self.removedDummies, self.n_dummy_conflicts = self.updateDummies(self.n_dummy_conflicts, self.addedCandidates, self.removedCandidates)

        self.addedTips = getComponentTips(self.addedComponents, self.candidate_graphs)
        self.removedTips = getComponentTips(self.removedComponents, self.candidate_graphs)

        # Calcule difference in cost items
        self.cost_items_difference.extract(self.candInfo, 
                                            self.candidate_graphs,
                                            self.curvature_offset_dict,
                                            self.ref_value_offset_dict,
                                            self.addedCandidates, 
                                            self.removedCandidates, 
                                            self.addedComponents, 
                                            self.removedComponents, 
                                            self.addedDummies, 
                                            self.removedDummies) 

        # Update to new cost items
        self.cost_items = self.cost_items + self.cost_items_difference

        if not self.isvalid():
            return False
        if not self.hasTwoTips():
            return False
        else:
            return True


    def isvalid(self):
        """
        Tests if candidates in a new connected component overlap
        """
        for c in self.addedComponents:
            if self.selfintersect(c):
                #print "Invalid component: "+str(c)
                return False
        return True

    def hasTwoTips(self):
        for item in self.addedTips:
            if len(item) != 2:
                return False

        #for item in self.changes['removedTips']:
        #    if len(item) != 2:
        #        return False

        return True

    def selfintersect(self, componentPath):
        """
        Tests if candidates overlap
        """
        lenPath = len(componentPath)
        for i in range(lenPath):
            for j in range(i+2,lenPath):
                if componentPath[j] > componentPath[i]:
                    if componentPath[j] in self.adjacencyList[componentPath[i]]:
                        return True
                elif componentPath[i] in self.adjacencyList[componentPath[j]]:
                    return True
        return False

    def getChanges(self):

        return self.addedTips, self.removedTips, self.addedCandidates, self.removedCandidates, self.addedComponents, self.removedComponents, self.addedDummies, self.removedDummies

    def reverseChanges(self):

        # Reverse changes

        self.updateBinaryList(self.removedCandidates, self.addedCandidates)

        for c in self.addedCandidates:
            self.components.removeVertex(c)
        for c in self.removedCandidates:
            self.components.addVertex(c)

        _, _, self.n_dummy_conflicts = self.updateDummies(self.n_dummy_conflicts, self.removedCandidates, self.addedCandidates)

        self.cost_items = self.cost_items - self.cost_items_difference

        # Reset tracked changes
        self.addedTips = []
        self.removedTips = []
        self.removedCandidates = []
        self.addedCandidates = []
        self.removedComponents = []
        self.addedComponents = []
        self.addedDummies = []
        self.removedDummies = []

        self.cost_items_difference = CostItemDifference

    def getChangesFromBinaryList(self, pos):

        if self.binaryList[pos] == 1: # If candidate is already in solution
            cand_add = []           # Add none
            cand_remove = [pos]     # Remove 
        else:
            cand_add = [pos]
            c = self.conflictList[pos]
            cand_remove = c[np.where(self.binaryList[c])]
             
        return cand_add, cand_remove

    def updateBinaryList(self, add, remove):
        self.binaryList[add] = 1
        self.binaryList[remove] = 0


    def updateDummies(self, n_dummy_conflicts, cand_add, cand_remove):       
        # For each removed candidate, dummies have to be added

        n_dummy_conflicts_copy = np.array(n_dummy_conflicts)

        dum_add = []
        for c in cand_remove:                                       # For each removed candidate
            dummy_ids = self.dummyConflictsList[c]
            n_dummy_conflicts_copy[dummy_ids] -= 1                  # Reduce number of conflicting candidates for this dummy
            ids = np.where(n_dummy_conflicts_copy[dummy_ids]==0)[0]  # If no more conflicts add a dummy
            dum_add.append(dummy_ids[ids])

        dum_add = list(chain(*dum_add))


        # For each added candidate, dummies have to be removed
        dum_remove = []
        for c in cand_add:       
            dummy_ids = self.dummyConflictsList[c]
            ids = np.where(n_dummy_conflicts_copy[dummy_ids]==0)[0]
            dum_remove.append(dummy_ids[ids])
            n_dummy_conflicts_copy[dummy_ids] += 1
        
        dum_remove = list(chain(*dum_remove))

        return dum_add, dum_remove, n_dummy_conflicts_copy

def getComponentTips(components, candidateGraphs):
    tips = [] 
    # print "get_component_tips:"
    for candidates in components:
        g = graphs.Candidate_Graph([])
        for candidate_id in candidates:
            g.merge(candidateGraphs[candidate_id])
        tips.append(g.all_degree_one())
    return tips


class CoolingScheduleMaker:

    def __init__(self, initialState, costFunction, initialProb=0.95, finalProb=0.0001):
        """
        Class to determine cooling schedule based on simulated solution
        """
        self.initialProb = initialProb
        self.finalProb = finalProb
        self.state = initialState                    # State object
        #self.temperatureLevels = temperatureLevels  # Number of temperatre levels
        self.deltaCostArray = []                    # Holds upward changed costs of simulation
        self.costArray = []                         # Holds all costs
        self.subCostArray = []
        self.costFunction = costFunction

    def simulate(self, n):
        """
        Creates neighbors n times start at given solution
        """
        nPaths = len(self.state.binaryList)

        for _ in range(n):

            # Get position (component id) to be changed
            position = random.randint(0,nPaths-1)

            # Change current solution
            isvalid = self.state.neighbor(position)

            # If is invalid reverse and skip
            if not isvalid:
                self.state.reverseChanges()
                continue
                
            # Calculate cost for current state
            metrics = self.costFunction.calculateMetrics(self.state)
            cost = self.costFunction.calculateCost(metrics)

            self.costArray.append(cost)
            self.subCostArray.append(metrics)

    def recalculateCost(self):
        """
        Recalculates costs with normalized sub costs
        """
        self.costArray = []
        for metrics in self.subCostArray:
            metricsNorm = self.costFunction.normalizeMetrics(metrics)
            cost = self.costFunction.calculateCost(metricsNorm)
            self.costArray.append(cost)

    def calculateUpwardCosts(self):
        self.deltaCostArray = []
        previous_cost = self.costArray[0]
        for current_cost in self.costArray:
            if current_cost > previous_cost:
                self.deltaCostArray.append(current_cost-previous_cost)
            previous_cost = current_cost

    def normalization(self):
        """
        Determines average values of sub cost for normalization of cost.
        Output: numpy array with 3 normalization values (float)
        """
        normArray = 1. / np.median(self.subCostArray,0)
        normArray[normArray == np.inf] = 0.0
        return normArray

    def averageDeltaCost(self):
        """
        Calculates average increasing cost
        """
        return np.mean(self.deltaCostArray)

    def initialCost(self):
        """
        Calculates average initial cost
        """
        return np.mean(self.costArray)
        

    def getInitialTemp(self):
        """
        Calculates initial temperature
        """
        t = - self.averageDeltaCost() / (self.initialCost() * np.log(self.initialProb))
        return t

    def getAlpha(self,initialTemp,nIterations):
        """
        Calculate the cooling rate
        """
        return (-self.averageDeltaCost() / (initialTemp * np.log(self.finalProb) * self.initialCost()))**(1.0/nIterations)

    def getFinalTemp(self, initialTemp, alpha, nIterations):
        """
        Calculates final temperature
        """
        return initialTemp*(alpha**nIterations)


def matrix_to_list(adjmat):
    # converts adjacency matrix to adjaceny list
    graph = [[] for v in adjmat]

    for i, v in enumerate(adjmat, 0):
        for j, u in enumerate(v, 0):
            if u != 0:
                #edges.add(frozenset([i, j]))
                graph[i].append(j)

    return [np.array(v,dtype=int) for v in graph]

class CandidateInformation:
    def __init__(self):
        
        # Path of each candidate
        self.paths = []

        # Dummie values
        self.dummy_strain = np.array([])
        self.dummy_lengths = np.array([])
        self.dummy_median_min_distance = 0.0
        self.dummy_median_max_distance = 0.0
        self.dummy_max_max_distance = 0.0

        # Candidate values
        self.strain = np.array([])
        self.min_distance = np.array([])
        self.max_distance = np.array([])
        self.min_reference_strain = np.array([])

        # Segment distance to root
        self.minDistToEdge = {}


class CostItems:
    def __init__(   self, 
                    sum_strain_roothair=0.0,\
                    sum_strain_dummy=0.0, \
                    sum_min_reference_strain=0.0, \
                    sum_length_dummy=0.0, \
                    sum_length_all=0.0, \
                    sum_min_distance_roothair=0.0, \
                    num_roothair=0, \
                    sum_max_distance_roothair=0.0):

        self.sum_strain_roothair = sum_strain_roothair
        self.sum_strain_dummy = sum_strain_dummy
        self.sum_min_reference_strain = sum_min_reference_strain

        # Total length of remaining dummies measure
        self.sum_length_dummy = sum_length_dummy
        self.sum_length_all = sum_length_all

        # Min distance to root
        self.sum_min_distance_roothair = sum_min_distance_roothair
        self.num_roothair = num_roothair
        
        # Max distance to root
        self.sum_max_distance_roothair = sum_max_distance_roothair

    def __add__(self, other):

        sum_strain_roothair = self.sum_strain_roothair + other.sum_strain_roothair
        sum_strain_dummy = self.sum_strain_dummy + other.sum_strain_dummy
        sum_min_reference_strain = self.sum_min_reference_strain + other.sum_min_reference_strain

        # Total length of remaining dummies measure
        sum_length_dummy = self.sum_length_dummy + other.sum_length_dummy
        sum_length_all = self.sum_length_all + other.sum_length_all
        
        # Min distance to root
        sum_min_distance_roothair = self.sum_min_distance_roothair + other.sum_min_distance_roothair
        num_roothair = self.num_roothair + other.num_roothair
        
        # Max distance to root
        sum_max_distance_roothair = self.sum_max_distance_roothair + other.sum_max_distance_roothair

        return CostItems(sum_strain_roothair, sum_strain_dummy, sum_min_reference_strain, \
                            sum_length_dummy, sum_length_all, 
                            sum_min_distance_roothair, num_roothair, sum_max_distance_roothair)

    def __sub__(self, other):

        sum_strain_roothair = self.sum_strain_roothair - other.sum_strain_roothair
        sum_strain_dummy = self.sum_strain_dummy - other.sum_strain_dummy
        sum_min_reference_strain = self.sum_min_reference_strain - other.sum_min_reference_strain

        # Total length of remaining dummies measure
        sum_length_dummy = self.sum_length_dummy - other.sum_length_dummy
        sum_length_all = self.sum_length_all - other.sum_length_all
        
        # Min distance to root
        sum_min_distance_roothair = self.sum_min_distance_roothair - other.sum_min_distance_roothair
        num_roothair = self.num_roothair - other.num_roothair
        
        # Max distance to root
        sum_max_distance_roothair = self.sum_max_distance_roothair - other.sum_max_distance_roothair

        return CostItems(sum_strain_roothair, sum_strain_dummy, sum_min_reference_strain, \
                            sum_length_dummy, sum_length_all, 
                            sum_min_distance_roothair, num_roothair, sum_max_distance_roothair)

class CostItemDifference(CostItems):

    def extract(self, candInfo, candidate_graphs, curvature_offset_dict, ref_value_offset_dict, cand_add, cand_remove, comp_add, comp_remove, dum_add, dum_remove):
        
        # Compute offsets for merged/unmerged components
        strain_offset = 0.
        reference_offset = 0.
        for c in comp_remove:
            if len(c)>1:
                for first, second in zip(c, c[1:]):
                    strain_offset += curvature_offset_dict[(min(first, second),max(first, second))]
                    reference_offset += ref_value_offset_dict[(min(first, second),max(first, second))]

        for c in comp_add:
            if len(c)>1:
                for first, second in zip(c, c[1:]):
                    strain_offset -= curvature_offset_dict[(min(first, second),max(first, second))]
                    reference_offset -= ref_value_offset_dict[(min(first, second),max(first, second))]
      
        # Curvature measure
        self.sum_strain_roothair = sum(candInfo.strain[cand_add]) \
                                        - sum(candInfo.strain[cand_remove]) \
                                            + strain_offset

        self.sum_min_reference_strain = sum(candInfo.min_reference_strain[cand_add]) \
                                        - sum(candInfo.min_reference_strain[cand_remove]) \
                                            + reference_offset
        
        # Total length of remaining dummies measure
        self.sum_length_dummy = sum(candInfo.dummy_lengths[dum_add]) \
                                        - sum(candInfo.dummy_lengths[dum_remove])

        self.sum_strain_dummy = sum(candInfo.dummy_strain[dum_add]) \
                                        - sum(candInfo.dummy_strain[dum_remove])
        
        # Number of components
        self.num_roothair = len(comp_add) - len(comp_remove)

        # Get tips of component
        tips_remove = getComponentTips(comp_remove, candidate_graphs)
        tips_add = getComponentTips(comp_add, candidate_graphs)

        # Sum min/max distances of tips
        min_distance = 0.0
        max_distance = 0.0

        for tip_pair in tips_add:
            distances = [candInfo.minDistToEdge[t] for t in tip_pair]
            if len(distances) == 2:
                min_distance += min(distances)
                max_distance += max(distances)
            else:
                #print "for tip_pair in tips_add: len(distances) = ", len(distances)
                return False

        for tip_pair in tips_remove:
            distances = [candInfo.minDistToEdge[t] for t in tip_pair]
            if len(distances) == 2:
                min_distance -= min(distances)
                max_distance -= max(distances)
            else:
                #print "for tip_pair in tips_remove: len(distances) = ", len(distances)
                return False

        # Min distance to root
        self.sum_min_distance_roothair = min_distance
        
        # Max distance to root
        self.sum_max_distance_roothair = max_distance

        return True

class Cost:
    def __init__(self, measure, cost_type, weights=[1., 1., 1.], normValues=np.array([1., 1., 1.])):
        self.measure = measure
        self.cost_type = cost_type
        self.normValues = normValues
        self.weights = np.array(weights)
        sum_weights = np.sum(weights)
        self.weights = np.float_(weights)/sum_weights
    
    def setNormValues(self, newValues):

        self.normValues = newValues

    def calculateMetrics(self, state):
        """
        Calculates metrics of state
        """

        cost_items = state.cost_items

        curvature_measure = (cost_items.sum_strain_roothair - cost_items.sum_min_reference_strain) #\
            #/ (cost_items.sum_min_reference_strain)

        tot_len_measure = cost_items.sum_length_dummy / cost_items.sum_length_all

        if cost_items.num_roothair > 0:
            min_dist_measure = cost_items.sum_min_distance_roothair / cost_items.num_roothair
        else:
            min_dist_measure = 0.0

        return np.array([curvature_measure, tot_len_measure, min_dist_measure])
    
    def normalizeMetrics(self, metrics):
        """
        Returns metrics normalized with Cost.normValues
        """
        return metrics * self.normValues

    def calculateCost(self, metrics):
        """
        Calculates cost from metrics using cost function settings
        """

        if self.cost_type == 'exp':
            return np.sum(self.weights * np.exp(metrics))
        elif self.cost_type == 'mean':
            return np.sum(self.weights * metrics)
        elif self.cost_type == 'rms':
            return weighted_root_mean_square(metrics, self.weights)
        elif self.cost_type ==  'pow3':
            return np.mean(metrics**3.)**(1/3.0)
        elif self.cost_type == 'pow4':
            return np.mean(metrics**4.)**(1/4.0)
        elif self.cost_type == 'geom':
            return gmean(metrics)
        else:
            return None



def weighted_root_mean_square(arr, weights):
    '''
    Get the root mean square value of the array values
    '''
    return np.sqrt(np.sum(weights * np.array(arr)**2.))