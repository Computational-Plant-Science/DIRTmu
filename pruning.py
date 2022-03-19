# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 13:11:49 2018

@author: pp34747
"""
import numpy as np
from skimage.measure import label
import skimage.filters.rank as rank


def numOfNeighbours(arr, a):
    # For each pixel==True in arr get number of its neighbours==True,
    # given the connectivity matrx a
    arrBinary = np.zeros_like(arr, dtype='uint8')   # Use 'uint8' because rank.sum give wrong values for the boolean arr
    arrBinary[np.where(arr)] = 1
    nneighbours = rank.sum(arrBinary, a)            # sum of neighbouring pixels
    #nneighbours = nneighbours * arrBinary           # apply mask. Note: We are only interestd in neighbours on the medial axis
    return nneighbours
    
def calcDistance(xy1,xy2):
    return np.sqrt(float((xy1[0]-xy2[0])*(xy1[0]-xy2[0])+(xy1[1]-xy2[1])*(xy1[1]-xy2[1])))



def findSegmentPixels(subsegments):
    """
    Pixels of each subsegment
    """
    # segmentPixels: for each subsegment i segmentPixels[i] contains all pixels of subsegment i
    nSubsegments = np.amax(subsegments)+1 # because Set(subsegments) = [0...nSubsegment-1]
    segmentPixels = np.empty(nSubsegments,dtype=list)
    pixels = zip(*np.where(subsegments>-1))
    for p in pixels:
        l = subsegments[p]
        if segmentPixels[l]  is None:
            segmentPixels[l] = [p]
        else:
            segmentPixels[l].append(p)
    return segmentPixels
    
def findSegmentType(subsegments,nneighbours):
    """
    Type of each subsegment
    """
    # segmentType: Type of each segment; 1 = tip; 2 = branch; 3 = junction
    nSubsegments = np.amax(subsegments)+1 # because Set(subsegments) = [0...nSubsegment-1]
    segmentType = np.empty(nSubsegments,dtype='uint8')
    pixels = zip(*np.where(subsegments>-1))
    for p in pixels:
        l = subsegments[p]
        segmentType[l] = nneighbours[p]
    return segmentType


def connectivity(subsegments,segmentPixels,data):
    # CONNECTIVITY BETWEEN SUBSEGMENTS
    # adjList: for each subsegment i adjList[i] contains id of adjacent subsegments
    nSubsegments = np.amax(subsegments)+1 # because Set(subsegments) = [0...nSubsegment-1]
    adjList = np.empty(nSubsegments,dtype=list)
    adjMat = np.zeros([nSubsegments,nSubsegments],dtype='uint8') # Initialize matrix of adjency

    for idx, pixels in enumerate(segmentPixels): # For all subsegments
        neighbourSegment = []
        for p in pixels:                    # For all pixels in current subsegment
            for dx in range(-1,2):          # Loop through adjacent pixels in image
                for dy in range(-1,2):
                    x = p[0]+dx
                    y = p[1]+dy
                    if x >= 0 and x<data.shape[0] and y >= 0 and y<data.shape[1]: # If pixel is inside image boundaries
                        if subsegments[x,y]!=-1 and subsegments[x,y]!=subsegments[p]: # If adjacent pixel is part of a different subsegent, save it
                            neighbourSegment.append(subsegments[x,y])
        adjList[idx] = list(set(neighbourSegment)) # only unique values in adjency list

        # Fill adjency matrix
        for n1 in neighbourSegment:
            adjMat[idx,n1] = 1
            adjMat[n1,idx] = 1

    return adjList, adjMat



    
def branchExtension(adjList,segmentType,segmentPixels):
    # Extends tip branches in segmentPixels with tips and junctions

    #branchIDs = np.where(np.logical_and(segmentType==2,nTips==1))[0]
    branchIDs = np.where(segmentType==2)[0]

    for idx, branchID in enumerate(branchIDs):
        for idy, neighbour in enumerate(adjList[branchID]):
            if segmentType[neighbour] == 1: # neighbour is a tip
                # add pixel to branch
                pixelsHere = segmentPixels[branchID]
                pixelsNew = segmentPixels[neighbour]
                pixelsHere.extend(pixelsNew)
                segmentPixels[branchID] = pixelsHere
            elif segmentType[neighbour] == 3:  # neighbour is a junction
                # add pixels to branch pixels
                pixelsHere = segmentPixels[branchID]
                pixelsNew = segmentPixels[neighbour]
                pixelsHere.extend(pixelsNew)
                segmentPixels[branchID] = pixelsHere


def branchConnectivity(adjList,segmentType,segmentPixels):
    # Adjency for the actual branches

    branchIDs = np.where(segmentType==2)[0]

    adjListBranch = np.empty(len(branchIDs),dtype=list)

    for idx, branchID in enumerate(branchIDs):
        neighbourSegment = []
        for idy, neighbour in enumerate(adjList[branchID]):
            if segmentType[neighbour] == 1: # neighbour is a tip
                # add pixel to branch
                pixelsHere = segmentPixels[branchID]
                #pixelsNew = segmentPixels[neighbour]
                #pixelsHere.extend(pixelsNew)
                #segmentPixels[branchID] = pixelsHere
            elif segmentType[neighbour] == 3:  # neighbour is a junction
                # add pixels to branch pixels
                #pixelsHere = segmentPixels[branchID]
                #pixelsNew = segmentPixels[neighbour]
                #pixelsHere.extend(pixelsNew)
                #segmentPixels[branchID] = pixelsHere
                for idz, adjBranches in enumerate(adjList[neighbour]):
                    if adjBranches!=branchID:
                        neighbourSegment.append(adjBranches)
            adjListBranch[idx] = list(set(neighbourSegment))
    return adjListBranch, branchIDs


def nNeighbourType(adjList,segmentType):
    nTips = np.zeros([len(adjList),],dtype='uint8')
    nBranches = np.zeros([len(adjList),],dtype='uint8')
    nJunctions = np.zeros([len(adjList),],dtype='uint8')

    for idx, adjSegment in enumerate(adjList):
        for idy, neighbour in enumerate(adjSegment):
            if segmentType[neighbour] == 1: # neighbour is a tip
                nTips[idx] = nTips[idx] + 1
            elif segmentType[neighbour] == 2: # neighbour is a branch
                nBranches[idx] = nBranches[idx] + 1
            elif segmentType[neighbour] == 3:  # neighbour is a junction
                nJunctions[idx] = nJunctions[idx] + 1
    return nTips, nBranches, nJunctions


def findSingleTips(segmentType,nBranches):
    singleTips = np.logical_and(segmentType==1,nBranches==0)
    return singleTips

def findWrongBranches(segmentType,nJunctions,nTips):
    wrongBranches = np.logical_and(segmentType==2,nJunctions==1)
    # TODO: DEBUG large loops get deleted
    wrongBranches = np.logical_and(wrongBranches,nTips==0)
    return wrongBranches

def findShortBranches(segmentType,adjList,nTips,segmentPixels,nneighbours,distance):

    shortBranches = np.zeros([len(adjList),],dtype='uint8')

    tipBranches = np.where(np.logical_and(segmentType==2,nTips==1))[0]      # Segments (type=2) that have a single tip (type=1) neighbor

    for idx, tipBranchID in enumerate(tipBranches):
        withinDistance = np.zeros([len(segmentPixels[tipBranchID]),],dtype='uint8')
        for idy, pixelJunction in enumerate(segmentPixels[tipBranchID]):
            if nneighbours[pixelJunction] > 2:
                for idz, pixelOther in enumerate(segmentPixels[tipBranchID]):
                    if idy==idz:
                        withinDistance[idz] = 1
                    elif distance[pixelOther] + calcDistance(pixelJunction,pixelOther) - distance[pixelJunction] < 3.0 * (distance[pixelJunction]-distance[pixelOther]+1.0)/calcDistance(pixelJunction,pixelOther) :
                        withinDistance[idz] = 1
        if np.all(withinDistance):
            shortBranches[tipBranchID]=1
    return shortBranches


def cleanSegments(skel,segmentsToClean,segmentPixels,nneighbours):
    changed = False

    for idx, tipBranchID in enumerate(segmentsToClean):
        for idy, pixels in enumerate(segmentPixels[tipBranchID]):
            if nneighbours[pixels] < 3:
                    skel[pixels] = 0
                    changed = True
    return changed

    
def prune(skel,distance):
    a = np.array([[1,1,1],
              [1,0,1],
              [1,1,1]])

    changed = True
    while changed:
        print("cleaning up tips...")
    
        # number of neighbours on medial axis
        nneighbours = numOfNeighbours(skel,a)
        nneighbours = nneighbours * skel
        
        # Segment medial eaxis into branches
        nneighbours[np.where(nneighbours>2)] = 3 #set all junctions to 3
        segmentLabels = label(nneighbours,connectivity=2)-1 # -1=no label; segement labels = [0...n]
    
        # Get segment type and its pixels
        segmentPixels = findSegmentPixels(segmentLabels)
        segmentType = findSegmentType(segmentLabels,nneighbours)
        
        # Adjency of segments
        adjList, adjMat = connectivity(segmentLabels,segmentPixels,skel)
        
        # number of adjacent neighbour type
        nTips, nBranches, nJunctions = nNeighbourType(adjList,segmentType)
    
        # add tip and junction pixels to banches
        branchExtension(adjList,segmentType,segmentPixels)
    
        # Find short branches with tips, single tips and wrong branches
        shortBranches = findShortBranches(segmentType,adjList,nTips,segmentPixels,nneighbours,distance) # is branch & has tip neighbour & and distance(tip,junction) is smaller than skeleton diameter
        singleTips = findSingleTips(segmentType,nBranches) # is tip & has no branch neighbours
        wrongBranches = findWrongBranches(segmentType,nJunctions,nTips)
    
        # Clean up
        segmentsToClean = np.logical_or(shortBranches==1,singleTips==1)
        segmentsToClean = np.where(np.logical_or(segmentsToClean,wrongBranches))[0]
        changed = cleanSegments(skel,segmentsToClean,segmentPixels,nneighbours)

    
    
    # Update all information
    nneighbours = numOfNeighbours(skel,a)
    nneighbours = nneighbours * skel
    nneighbours[np.where(nneighbours>2)] = 3 #set all junctions to 3
    segmentLabels = label(nneighbours,connectivity=2)-1 # -1=no label; segement labels = [0...n]
    segmentPixels = findSegmentPixels(segmentLabels)
    segmentType = findSegmentType(segmentLabels,nneighbours)
    adjList, adjMat = connectivity(segmentLabels,segmentPixels,skel)
    nTips, nBranches, nJunctions = nNeighbourType(adjList,segmentType)
    adjListBranch, branchIDs = branchConnectivity(adjList,segmentType,segmentPixels)