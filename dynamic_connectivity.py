#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class ConnectedComponents:
    """
    From a set of known vertices and edges (merge_list)
    adds or removes vertices to graph and keeps track of 
    connected components dynamically.
    """ 
    def __init__(self, merge_list):
        """
        Initialize graph and connected components with merge_list.
        All vertices in graph are filtered and connected components 
        are empty.
        merge_list = list of list with integer ids of vertices
        e.g. merge_list = [[1,2,3],[0],[0],[0]]
        """

        self.merge_list = merge_list

        # Set component key to minimum vertex value in component
        self.components = {}
        self.vertices = {}
        
        n_vertices = len(self.merge_list)

    def addVertex(self, vertex):

        if vertex in self.vertices:

            comp_add = {}
            comp_remove = {}

        else:

            # Find all adjacent components in current solution
            other_components = set()
            for neighbor in self.merge_list[vertex]: #TODO: Better get neighbors from filtered graph?
                if neighbor in self.vertices:
                    oc = self.vertices[neighbor]
                    other_components.add(oc)
            
            # Not connected to other components
            # -> Create new component
            if len(other_components) == 0:

                # New component id
                new_id = vertex

                # Add new vertex
                self.vertices[vertex] = new_id
                self.components[new_id] = [vertex]
                
                comp_add = {new_id: [vertex]}
                comp_remove = {}

            # Connected to other components
            else:

                # New component id
                new_id = min(other_components)
                new_id = min(new_id, vertex)
                
                # Get all vertices and remove old component
                comp_remove = {}
                new_v_list = [vertex]
                for c in other_components:
                    v_list = self.components.pop(c)
                    comp_remove[c] = list(v_list) # save copy of original list

                    if len(new_v_list) == 1: # Fist iteration, i.e. new_v_list == [vertex]
                        if v_list[-1] in self.merge_list[vertex]:
                            new_v_list = v_list + new_v_list
                        else:
                            new_v_list = v_list[::-1] + new_v_list # no need to copy? 
                    else:   # Second iteration, if any
                        if v_list[0] in self.merge_list[vertex]:
                            new_v_list = new_v_list + v_list
                        else:
                            new_v_list = new_v_list + v_list[::-1]

                
                for v in new_v_list:
                    self.vertices[v] = new_id

                self.components[new_id] = new_v_list

                comp_add = {new_id: list(new_v_list)}  # save copy of original list

        return comp_add, comp_remove
            

    def removeVertex(self, vertex):

        if vertex not in self.vertices:

            comp_add = {}
            comp_remove = {}

        else:
            # Get component id/ remove from vertices
            comp_id = self.vertices.pop(vertex)
            size_component = len(self.components[comp_id])

            # Just remove component
            if size_component == 1:

                self.components.pop(comp_id)

                comp_add = {}
                comp_remove = {comp_id: [vertex]}

            # Create new components
            else:
                # Get neighbors in graph (must be len=1 or len=2)
                v_list = self.components[comp_id]
                index = v_list.index(vertex)
                
                # If is terminal vertex
                # No need for finding connected component
                if index == 0 or index == size_component-1:
                    
                    comp_remove = {comp_id: list(self.components[comp_id])}

                    # Remove vertex from component
                    self.components[comp_id].remove(vertex)
                    new_comp_id = min(self.components[comp_id])
                    self.components[new_comp_id] = self.components.pop(comp_id)
                    
                    for v in self.components[new_comp_id]:
                        self.vertices[v] = new_comp_id
                
                    comp_add = {new_comp_id: list(self.components[new_comp_id])}

                # Else find new connected components
                else:
                    
                    comp_remove = {comp_id: list(self.components[comp_id])}
                    comp_add = {}
                    self.components.pop(comp_id)
                    
                    # Left component
                    v_ids = v_list[0:index]
                    new_comp_id = min(v_ids)
                    self.components[new_comp_id] = v_ids
                    for v in v_ids:
                        self.vertices[v] = new_comp_id
                    comp_add[new_comp_id] = list(v_ids)

                    # Right component
                    v_ids = v_list[index+1:]
                    new_comp_id = min(v_ids)
                    self.components[new_comp_id] = v_ids
                    for v in v_ids:
                        self.vertices[v] = new_comp_id
                    comp_add[new_comp_id] = list(v_ids)


        return comp_add, comp_remove

if __name__ == '__main__':
    merge_list = [[1],[0,2],[1,3],[2,4],[3,5],[4],[7],[6],[]]

    cc = ConnectedComponents(merge_list)
    cc.addVertex(0)
    cc.addVertex(1)
    cc.addVertex(2)
    cc.addVertex(3)
    cc.addVertex(4)
    cc.addVertex(5)
    cc.addVertex(6)
    cc.addVertex(7)
    cc.addVertex(8)

    print "Graph:"
    print cc.components
    print cc.vertices

    print " "
    print "Remove vertex 2"
    comp_add, comp_remove = cc.removeVertex(2)
    print comp_add, comp_remove
    print cc.components
    print cc.vertices

    print " "
    print "Remove vertex 6"
    comp_add, comp_remove = cc.removeVertex(6)
    print comp_add, comp_remove
    print cc.components
    print cc.vertices

    print " "
    print "Remove vertex 8"
    comp_add, comp_remove = cc.removeVertex(8)
    print comp_add, comp_remove
    print cc.components
    print cc.vertices

    print " "
    print "Add vertex 2"
    comp_add, comp_remove = cc.addVertex(2)
    print comp_add, comp_remove
    print cc.components
    print cc.vertices