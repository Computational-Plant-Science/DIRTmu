#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""
import math

import numpy as np
from skimage import measure
from skimage.draw import disk as drawDisk
from skimage.morphology import medial_axis, opening, closing, disk
from skimage.transform import resize
from skimage.measure import regionprops, label
from scipy.interpolate import splprep, splev
from sklearn.neighbors import NearestNeighbors

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt




def extractDiameter(rootImg,rootIdx):
    print("Computing radius of the main root")
    medialAxisImg = selectLargestComponent(rootImg).astype(dtype=float)

    s=medialAxisImg.shape
    medialAxisImgSmall=resize(medialAxisImg.copy(),(100,int(float(s[1])/float(s[0])*100)), anti_aliasing=True, mode='reflect') # Downsampling
    #medialAxisImgSmall = medialAxisImgSmall > 0.5

    # Compute the medial axis (skeleton) and the distance transform for the root object
    medialAxisImg=closing(medialAxisImg,disk(3))
    medialAxisImg=opening(medialAxisImg,disk(3))
    medialAxisImgSmall=closing(medialAxisImgSmall,disk(1))
    medialAxisImgSmall=opening(medialAxisImgSmall,disk(1))
    skel, distance = medial_axis(medialAxisImg, return_distance=True)
    skelSmall, distanceSmall = medial_axis(medialAxisImgSmall, return_distance=True)
    dist_on_skel = distance * skel
    dist_on_skelSmall = resize(distanceSmall * skelSmall,s, anti_aliasing=False, mode='reflect') # Sampling back to orginal size
    
    skelList=np.where(dist_on_skelSmall==0)
    dist_on_skel[skelList]=0
        
    dist_on_skel=pruneMA(dist_on_skel)
    
    diameterPos=np.where(dist_on_skel>0)
    root_diameter = 2.*np.median(dist_on_skel[diameterPos])
    print("Root diameter = "+ str(root_diameter))
    
    return dist_on_skel, root_diameter
    
def pruneMA(MAimg):
    print('Pruning medial axis')
    # positions of the MA
    pos=np.where(MAimg>0)
    maxVal = np.max(MAimg)
    oneConnectedEnds=[]
    for i in zip(pos[0],pos[1]):
        if MAimg[i[0],i[1]] < maxVal/1.5:
            MAimg[i[0],i[1]]=0
    # Loop through MA
    # for i in zip(pos[0],pos[1]):
    #     count=0
    #     for j in (-1,0,1):
    #         for k in (-1,0,1): 
    #             try:
    #                 if MAimg[i[0]+j,i[1]+k] >0:
    #                     if j!=0 or k!=0: 
    #                         count+=1
    #             except: pass
    #     if count == 1:
    #         oneConnectedEnds.append(i)
    # collect=[]
    # for idx,i in enumerate(oneConnectedEnds):
    #     MAimg[i[0],i[1]]=0
    #     for j in (-1,0,1):
    #         for k in (-1,0,1):
    #             try:
    #                 if MAimg[i[0]+j,i[1]+k] >0:
    #                     if j!=0 or k!=0:
    #                         collect.append((i[0]+j,i[1]+k)) 
    #             except: pass
    #     if len(collect)==1:
    #         oneConnectedEnds.append(collect[0])
    #     collect=[]
   
    return MAimg
    
def reconstructRoot(MAimg):
    print('Reconstructing main root')
    pos=np.where(MAimg>0)
    imgShape = MAimg.shape
    rootImg=np.zeros(MAimg.shape, dtype=int)
    for i in zip(pos[0],pos[1]):
        radius =  MAimg[i[0],i[1]]
        rr_img, cc_img = drawDisk(i, radius, shape=imgShape)
        try:
            rootImg[rr_img, cc_img] = 255
        except: pass
    #for i in zip(pos[0],pos[1]):
    #    rootImg[i[0],i[1]] = 0
    return rootImg

def reconstructLatRoots(rootImg, mainRootImg):
    print('Reconstructing lateral roots')

    latRootImg = rootImg.copy()
    latRootImg[np.where(mainRootImg == 1)] = 0
    mainRootSize = np.count_nonzero(mainRootImg)
    #latMAImg=extractDiameter(latRootImg, 1)
    
    #latMAImg = reconstructRoot(latMAImg)
    latRootImg = opening(latRootImg,disk(1))
    segments = label(latRootImg, background=0)
    props = regionprops(segments)
    areas = [prop['area'] for prop in props]
    
    for prop in props:
        
        if prop['area'] < mainRootSize*0.001:
            coords = list(zip(*prop['coords']))
            latRootImg[coords[0], coords[1]] = 0

    centroid = [prop['centroid'] for prop in props]
    #thresh = filters.threshold_minimum(np.array(areas))

    return latRootImg

def selectLargestComponent(img):
    """
    Returns the largest component of the image
    https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image
    """
    labels  = label(img, background=0)
    assert( labels.max() != 0 ) # assume at least 1 connected component
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    
    return largestCC.astype(int)

def contour_overlap(countours, refContours):
    #indices = []
    refPoints = []
    overlaps = []

    refPoints = [pt for sublist in refContours for pt in sublist]

    if len(refPoints) > 0:
        nbrs = NearestNeighbors(n_neighbors=1)
        nbrs.fit(refPoints)        
        for contour in countours:
            t = []
            dist, _ = nbrs.kneighbors(contour)
            for i,d in enumerate(dist):
                if d < 1:
                    t.append(1)
                else:
                    t.append(0)
            overlaps.append(t)
    else:
        for contour in countours:
            overlaps.append([0]*len(contour))

    return overlaps    

    #for contour in countours:
    #    nbrs.kneighbors(...., return_distance=False)

def length(x, y):
    x_diff = np.diff(x)
    y_diff = np.diff(y)
    l = np.sum(np.sqrt(x_diff * x_diff + y_diff * y_diff))
    return l

def cumulativelength(x, y):
        cLength = [0.]
        for i in range(1,len(x)):
            cLength.append (cLength[i-1] +
                            math.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2))
        return cLength

def smooth(x, y):
    if len(x) <= 3:
        return x, y
    
    w = np.ones(len(x))
    w[0] = 1000
    w[-1] = 1000
    tck, u = splprep([x,y], w=w)
    new_points = splev(u, tck)
    return new_points

def classify_roothair(roothairs, contour):
    '''
    Computes the distance of the root hair end points to an edge
    '''
    allNearestContours = []
    allNearestPtIDs = []
    allDists = []

    #pts, segmentIDs = zip(*[(pt,ind) for (ind,seg) in enumerate(contour) for pt in seg])
    # Get points from all contours into single list + for each point its location in corresponding contour +list with indices of contour
    pts, idsOfPts, idsOfContours = list(zip(*[(pt,ptInd,segmentInd) for (segmentInd,segment) in enumerate(contour) for ptInd,pt in enumerate(segment)]))
    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(pts)

    for rh in roothairs:

        # distance to edge and indices (location in 'edge')
        first_point = [rh.x[0],rh.y[0]]
        last_point = [rh.x[-1],rh.y[-1]]
        end_points = [first_point, last_point]
        dists, indices = nbrs.kneighbors(end_points)

        minID = np.argmin(dists)
        nearestIndex = indices[minID][0]
        nearestDist = dists[minID][0]
        
        allNearestContours.append(idsOfContours[nearestIndex])
        allNearestPtIDs.append(idsOfPts[nearestIndex])
        allDists.append(nearestDist)

    return allNearestContours, allNearestPtIDs, allDists

def calculate_densities(contours,allNearestContours,allNearestPtIDs, windowSizePxl):

    all_densities_list = []

    for segmentIndex, contour in enumerate(contours):
        ids = np.where(np.array(allNearestContours)==segmentIndex)[0]
        if len(ids) == 0:
            continue
        ptIDs = sorted(np.array(allNearestPtIDs)[ids])
        cumLength = cumulativelength(contour[1],contour[0])
        current_densities = rollingSum(np.array(cumLength)[ptIDs], cumLength[0], cumLength[-1], windowSizePxl, 1)
        all_densities_list.extend(current_densities)

    return all_densities_list

def rollingSum(x, start, stop, windowSizePxl=1000, step=1):
    
    if stop < windowSizePxl:
        return []

    density_list = []

    for start in range(int(start), int(stop - windowSizePxl + step), int(step)):
        stop = start + windowSizePxl
        current_sum = 0
        for val in x:
            if val >= start and val < stop:
                current_sum = current_sum + 1
        density_list.append(current_sum)
    return density_list


def segmentRootComponents(labelImg, rootIdx):

    newRootIdx = 1

    # Extract root pixels to binary image
    rootImg=np.zeros_like(labelImg)
    rootImg[np.where(labelImg == rootIdx)] = newRootIdx #labelImg[np.where(labelImg == rootIdx)]
    #rootImg = selectLargestComponent(rootImg)  

    # Prune and reconstruct main root
    MAImg,root_diameter=extractDiameter(rootImg, newRootIdx)
    mainRootImg = reconstructRoot(MAImg)
    mainRootImg = selectLargestComponent(mainRootImg)

    # Reconstruct lateral roots
    latRootImg = reconstructLatRoots(rootImg, mainRootImg)
    #bdryLat = segmentation.find_boundaries(latRootImg, connectivity=2, mode='inner')

    return mainRootImg, latRootImg, root_diameter


def computeDensity(labelImg, roothair, rootIdx, pixel_size):

    print('DIRT/mu-Roothair density estimation')

    pixel_size = float(pixel_size) # pixel size in microns per pixel
    window_size = 1000 # sliding window size in microns (=1000)

    # Check if root exists. Else retrun NaN values
    if rootIdx not in labelImg:
        print("No root found in image! Root index dose not correspond to any pixel in image.")
        rhClass = [np.nan for val in roothair]
        rhPositions = [np.nan for val in roothair]
        results = {"RH Count Total":len(roothair), 
                "RH Count Bottom":np.nan,
                "RH Count Top":np.nan,
                "RH Count Max": np.nan,
                "RH Count Min": np.nan,
                "RH Count Mean": np.nan,
                "RH Count Std": np.nan,
                "Edge Length Total (mu):":np.nan,
                "Edge Length Bottom (mu)":np.nan,
                "Edge Length Top (mu)":np.nan,
                "Window size (mu)":window_size,
                "Root Diameter (mu)":np.nan}
        return results, rhClass, rhPositions, {"closestSegments":[np.nan], "edge_classes":[np.nan], "edge_segments":[np.nan], "edge_position":[np.nan]}

    mainRootImg, latRootImg, root_diameter = segmentRootComponents(labelImg, rootIdx)

    # Get main and lateral root contours
    mainContours = measure.find_contours(mainRootImg, 0.5)
    latContours = measure.find_contours(latRootImg, 0.5)

    # Order of length of main contours
    mainContoursSmooth = [smooth(seg[:,1],seg[:,0]) for seg in mainContours]
    contourLength = [length(seg[1],seg[0]) for seg in mainContoursSmooth]
    twoLongestIDs = np.argsort(contourLength)[-2:] # we will need only first two values 

    # Get top/bottom edge index
    median_y = [np.median(mainContours[ind][:,0]) for ind in twoLongestIDs]
    topID = twoLongestIDs[np.argmin(median_y)]      # note: image is upside down
    bottomID = twoLongestIDs[np.argmax(median_y)]   # note: image is upside down

    # Label continuous contour segments; non-overlapping main/lateral segments
    edge_segments = []  # holds edge segment
    edge_ids = []       # id of parent edge
    edge_classes = []   # 'main', 'lateral'
    edge_position = []  # 'top', 'bottom', 'other'
    all_contours = []   # holds contours from main and lateral roots

    # gets sub-segments of main root contour
    overlaps = contour_overlap(mainContours, latContours)
    for contourInd in range(len(mainContours)): # for each main contour
        contour = mainContours[contourInd]
        all_contours.append(contour)
        orvlp = overlaps[contourInd]
        labels = label(np.logical_not(orvlp))   # segmentation
        for i in np.arange(1,max(labels)+1):    # for each segmented contour part
            indices = np.where(labels==i)[0]
            edge_segments.append(contour[indices, :])
            edge_ids.append(contourInd)
            edge_classes.append('main')
            if contourInd == topID:
                edge_position.append('top')
            elif contourInd == bottomID:
                edge_position.append('bottom')
            else:
                edge_position.append('other')
            
    # gets sub-segments of lateral root contour
    overlaps = contour_overlap(latContours, mainContours)
    for contourInd in range(len(latContours)): # for each lateral contour
        contour = latContours[contourInd]
        all_contours.append(contour)
        orvlp = overlaps[contourInd]
        labels = label(np.logical_not(orvlp))   # segmentation
        for i in np.arange(1,max(labels)+1):
            indices = np.where(labels==i)[0]
            edge_segments.append(contour[indices, :])
            edge_ids.append(contourInd)
            edge_classes.append('lateral')
            edge_position.append('other')

    # Find closest contour segment for all roothairs 
    closestSegments, ptIndices, minDist = classify_roothair(roothair, edge_segments)
    rhClass = [edge_classes[val] for val in closestSegments] 
    rhPositions = [edge_position[val] for val in closestSegments]

    rhCount = np.bincount(closestSegments)

    edge_segments_smooth = [smooth(seg[:,1],seg[:,0]) for seg in edge_segments]
    segLength = [length(seg[1],seg[0]) for seg in edge_segments_smooth]

    mainIds = np.where(np.array(edge_classes)=='main')[0]
    edge_segments_main = [edge_segments_smooth[i] for i in mainIds]
    rolling_sum_densities = calculate_densities(edge_segments_main,closestSegments,ptIndices,float(window_size)/pixel_size)
    if len(rolling_sum_densities) > 0:
        rs_density_max = max(rolling_sum_densities)
        rs_density_min = min(rolling_sum_densities)
        rs_density_mean = np.mean(rolling_sum_densities)
        rs_density_std = np.std(rolling_sum_densities)
    else:
        rs_density_max = -1
        rs_density_min = -1
        rs_density_mean = -1
        rs_density_std = -1

    topCount = sum([val for ind,val in enumerate(rhCount) if edge_position[ind]=='top'])
    topLength = pixel_size * sum([val for ind,val in enumerate(segLength) if edge_position[ind]=='top'])
    bottomCount = sum([val for ind,val in enumerate(rhCount) if edge_position[ind]=='bottom'])
    bottomLength = pixel_size * sum([val for ind,val in enumerate(segLength) if edge_position[ind]=='bottom'])

    totCount = sum([val for ind,val in enumerate(rhCount) if edge_classes[ind]=='main'])
    totLength = pixel_size * sum([val for ind,val in enumerate(segLength) if edge_classes[ind]=='main'])

    #rh_length = [length(rh.x,rh.y) for rh in roothair]
    #rh_length_main = sum([val for ind,val in enumerate(rh_length) if edge_classes[closestSegments[ind]]=='main'])
    #rh_length_lateral = sum([val for ind,val in enumerate(rh_length) if edge_classes[closestSegments[ind]]=='lateral'])
    
    results = {"RH Count Total":totCount, 
                "RH Count Bottom":bottomCount,
                "RH Count Top":topCount,
                "RH Count Max": rs_density_max,
                "RH Count Min": rs_density_min,
                "RH Count Mean": rs_density_mean,
                "RH Count Std": rs_density_std,
                "Edge Length Total (mu):":totLength,
                "Edge Length Bottom (mu)":bottomLength,
                "Edge Length Top (mu)":topLength,
                "Window size (mu)":window_size,
                "Root Diameter (mu)":root_diameter*pixel_size}

    return results, rhClass, rhPositions, {"closestSegments":closestSegments, "edge_classes":edge_classes, "edge_segments":edge_segments, "edge_position":edge_position}

def plotDensity(roothair, labelImg, closestSegments, edge_classes, edge_segments, edge_position, output_path, linewidth=1):
    """
    Plots edges of root with corresponding root hairs
    """
    # Plotting    
    fig = Figure(figsize=(10, 10), dpi=300)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(labelImg, cmap='gray')

    # Plot root hair
    n_components = len(roothair)
    randOrder = np.arange(n_components)
    np.random.shuffle(randOrder)
    for counter, rh in enumerate(roothair):
        contourID = closestSegments[counter]
        if edge_classes[contourID]=='main':
            rgba = plt.cm.Spectral(float(np.clip(randOrder[counter], 0, n_components))/n_components)
            rgb = rgba[0:3]
            ax.plot(rh.y, rh.x, color=rgb, solid_capstyle='round', linewidth=linewidth)
        else:
            ax.plot(rh.y, rh.x, 'r', solid_capstyle='round', linewidth=linewidth)

    # Plot countour lines
    for ind, seg in enumerate(edge_segments):
        x, y = smooth(seg[:, 0], seg[:, 1])
        if edge_classes[ind]=='main':
            if edge_position[ind] == 'top':
                ax.plot(y, x, 'g', solid_capstyle='round', linewidth=linewidth)
            elif edge_position[ind] == 'bottom':
                ax.plot(y, x, 'b', solid_capstyle='round', linewidth=linewidth)
            else:
                ax.plot(y, x, 'm', solid_capstyle='round', linewidth=linewidth)
        else: 
            ax.plot(y, x, 'r', solid_capstyle='round', linewidth=linewidth)

    #filename, file_extension = os.path.splitext(filename)
    fig.savefig(output_path, dpi=600, bbox_inches='tight')