# -*- coding: utf-8 -*-
'''
----------------------------------------------------------------------------------------------------
This module preprocesses

----------------------------------------------------------------------------------------------------
'''

import segmentation
import pruning
import numpy as np
import skimage.morphology as morph
import skimage.filters.rank as rank
from sklearn.neighbors import NearestNeighbors

class Preprocessing:
    
    def __init__(self, id_root=1, id_background=2, id_roothair=3, is_close_gaps=True, is_remove_clusters=True, is_prune=True):
        self.id_root = id_root
        self.id_background = id_background
        self.id_roothair = id_roothair
        self.is_close_gaps = is_close_gaps
        self.is_remove_clusters = is_remove_clusters
        self.is_prune = is_prune
        
    def run(self,classes):
        '''
        Runs preprocessing pipeline
        '''
        # classes = self.transform(classes)

        # Close small gaps and remove small clusters
        if self.is_close_gaps:
            classes = self.close_gaps(classes, size=2)
        
        if self.is_remove_clusters:
            # Median root hair thickness
            rh_thickness = self.get_median_rh_thickness(classes,self.id_roothair)
            
            # Remove small root hair clusters
            if rh_thickness is None:
                min_size = None
            else:
                min_size = int(np.ceil((rh_thickness**2)))     # min number of pixels in clusters
            classes = self.remove_clusters(classes, 
                                               id_remove=self.id_roothair, size=min_size)
    
            # Remove small root clusters
            if rh_thickness is None:
                min_size = None
            else:
                min_size=int(np.ceil(((10*rh_thickness)**2))) # min number of pixels in clusters
            classes = self.remove_clusters(classes, 
                                               id_remove=self.id_root, size=min_size)
     
        # Extract medial axis of root hairs
        skel, distance = self.get_ma(classes, self.id_root, self.id_roothair)
        
        if self.is_prune:
            # Prune medial axis
            skel, distance = self.prune(skel, distance)
        
        # Distance on skel
        dist_on_skel = distance * skel
        
        # Distance of medial axis to main root
        dist_to_root = self.get_dist_to_root(classes,skel,self.id_root)
        
        self.skel, self.dist_on_skel, self.dist_to_root = skel, dist_on_skel, dist_to_root        
        return skel, dist_on_skel, dist_to_root
    
    def out(self,path):
        pass

    def close_gaps(self, classes, size=1):
        '''
        Closes small gaps using binary closing and removes small clusters using
        binary opening.
        '''
        # Close gaps in root hairs
        data_RH = np.zeros_like(classes)
        data_RH[np.where(classes == self.id_roothair)] = 1
        data_RH = morph.binary_closing(data_RH, morph.disk(size))
        data_RH = morph.binary_opening(data_RH, morph.disk(size))
        
        # Close gaps in root
        data_R = np.zeros_like(classes)
        data_R[np.where(classes == self.id_root)] = 1
        data_R = morph.binary_closing(data_R, morph.disk(size))
        data_R = morph.binary_opening(data_R, morph.disk(size))
        
        # Close gaps in background
        data_BG = np.zeros_like(classes)
        data_BG[np.where(classes == self.id_background)] = 1
        data_BG = morph.binary_closing(data_BG, morph.disk(size))
#        data_BG = morph.binary_opening(data_BG, morph.disk(size))
        
        # Merge all together into one array
        classes_new = np.zeros_like(classes)
        classes_new[np.where(data_BG)] = self.id_background
        classes_new[np.where(data_R)] = self.id_root
        classes_new[np.where(data_RH)] = self.id_roothair
        
        return classes_new

    def get_median_rh_thickness(self, classes, id_roothair):
        '''
        Computes the median thickness of root hairs
        '''
        data = np.zeros_like(classes)
        data[np.where(classes == id_roothair)] = 1 # where root hair
        skel, distance = morph.medial_axis(data, return_distance=True)
        dist_on_skel = distance * skel
        if sum(sum(dist_on_skel > 0)) > 0:
            median_dist = np.median(dist_on_skel[np.where(dist_on_skel > 0)])
            return 2.*median_dist
        else:
            return None


    def remove_clusters(self,classes,id_remove,size=10):
        '''
        Removes small clusters of class id_remove
        '''
        if size is None:
            return classes
        data = np.zeros_like(classes,dtype=bool)
        data[np.where(classes == id_remove)] = True
        data_clean = morph.remove_small_holes(data,min_size=size,connectivity=2)
        data_clean[np.where(classes != id_remove)] = True
        classes_new = data_clean*classes
        return classes_new

    def get_ma_temp(self,classes,id_root,id_roothair):
        '''
        Merge root hairs and root to get medial axis
        '''
        data = np.zeros_like(classes)
        data[np.where(classes == id_root)] = 1
        data[np.where(classes == id_roothair)] = 1
        labels, labels_num = morph.label(data, return_num=True, connectivity=1)
        # Compute the medial axis (skeleton) and the distance transform
        skel = np.zeros_like(data)
        distance = np.zeros_like(data)
        for l in range(labels_num):
            data_loc = np.zeros_like(data)
            data_loc[np.where(labels==l+1)] = 1
            skel_loc, distance_loc = morph.medial_axis(data_loc, return_distance=True)
            skel, distance = skel+skel_loc, distance+distance_loc
        
        # Clip medial axis of root (keep only root hairs)
        data = np.zeros_like(classes)
        data[np.where(classes == id_roothair)] = 1
        distance = distance * data
        skel = skel * data
        return skel, distance

    def get_ma(self,classes,id_root,id_roothair):
        '''
        Merge root hairs and root to get medial axis
        '''
        data = np.zeros_like(classes)
        data[np.where(classes == id_root)] = 1
        data[np.where(classes == id_roothair)] = 1

        # Compute the medial axis (skeleton) and the distance transform
        skel, distance = morph.medial_axis(data, return_distance=True)

        # Clip medial axis of root (keep only root hairs)
        data = np.zeros_like(classes)
        data[np.where(classes == id_roothair)] = 1
        #distance = distance * data
        skel = skel * data

        # Fill small holes in medial axis
        selem = np.array([  [0,1,0],
                            [1,0,1],
                            [0,1,0]])
        fill = rank.sum(skel.astype('int64'), selem) 
        skel[np.where(fill == 4)] = 1

        return skel, distance


    def prune(self,skel,distance):     
        '''
        Prunes medial axis
        '''
        pruning.prune(skel, distance)
        return skel, distance
        

    def get_dist_to_root(self,classes,skel,id_root):
        '''
        Computes the distance of the root hair medial axis to the main root
        '''
        edge = self.find_edge(classes, id_root)       # pixel coordinates of edge of main root
        skel_dist_to_edge = np.zeros(skel.shape)
        if len(edge) > 0:
            skelCoords = zip(*np.where(skel))   # pixel coordinates of medial axis
            nbrs = NearestNeighbors(n_neighbors=1)
            nbrs.fit(edge)
            # distance to edge and indices (location in 'edge')
            edgeDist, edgeIDs = nbrs.kneighbors(skelCoords)
            # REMOVES ROOT HAIR PIXELS IF TOO CLOSE TO MAIN ROOT + PLOTS
            skel_dist_to_edge = np.zeros(skel.shape,dtype=float)
            # Map edge distance onto 2D array for image
            for i, index in enumerate(edgeIDs):
                skel_dist_to_edge[skelCoords[i]] = edgeDist[i][0]
        return skel_dist_to_edge
        
    
    def find_edge(self,classes, class_id):
        '''
        Extracts the edge of class_id in the array classes
        '''
        edge_loc = np.zeros_like(classes)
        edge_loc[np.where(classes != class_id)] = 1
        edge_mask = np.zeros_like(classes)
        edge_mask[np.where(classes == class_id)] = 1
        arr = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]])
        n_neighbours_edge = segmentation.Segmentation.numOfNeighbours(edge_loc, arr)
        n_neighbours_edge = n_neighbours_edge * edge_mask
        edge = zip(*np.where(n_neighbours_edge > 0))
        return edge