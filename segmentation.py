# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 12:30:00 2018

@author: pp34747
"""
import numpy as np
import skimage.filters.rank as rank
from skimage.measure import label
import networkx as nx
import graph_tool.all as gt
import matplotlib.pyplot as plt


class Segmentation:
    def __init__(self, medialAxis, distance):
        self.medialAxis = medialAxis
        self.distance = distance
        self.nneighbours = None
        self.segmentIDs = None
        self.neighborSegments = None
        self.segmentPixels = None
        self.segmentType = None
        self.segments = None


        # SEGMENTATION OF MEDIAL AXIS
        # tips have 1 neighbor
        # branches have 2 neighbors
        # junctions have 3 or more neighbors
        a = np.array([[1,1,1],
                      [1,0,1],
                      [1,1,1]])

        self.nneighbours = Segmentation.numOfNeighbours(self.medialAxis,a) # 2D array: number of neighboring medial axis pixels
        self.nneighbours = self.nneighbours * self.medialAxis
        self.nneighbours[np.where(self.nneighbours>2)] = 3        # set all junctions to 3

        self.segmentIDs = label(self.nneighbours,connectivity=2)  # 2D array: con. components; IDs = [1...maxLabels]
        #TODO: Depreceated, use skimage.measure.regionprops(labels), coords
        self.findSegmentPixels()   # Dictionary: a list of pixels per segmentID
        self.findSegmentType() # Dictionary: type per segmentID
        self.createSegments()
        self.allSegmentNeighbors()

    def pixelNeighbors(self,x,y,shape):
         # Get position of neighboring pixels for x,y given the shape of the array
         neighbors = [(x2, y2)  for x2 in range(x-1, x+2)
                                for y2 in range(y-1, y+2)
                                if (-1 < x <= shape[0] and
                                    -1 < y <= shape[1] and
                                    (x != x2 or y != y2) and
                                    (0 <= x2 <= shape[0]) and
                                    (0 <= y2 <= shape[1]))]
         return neighbors

    def allSegmentNeighbors(self):
        for v in self.segments.values():
            v.neighbors = self.segmentNeighbors(v) # Get neighboring segments from labels
            v.setDirection()

    def segmentNeighbors(self, segm):
        # For a segment with ID segm.Label get all neighboring segments
        #segm = self.segments[segmentLabel]
        neighbourLabels = {}
        for p in segm.pixels:
            allneighbors = self.pixelNeighbors(p[0],p[1],self.segmentIDs.shape)
            segmentneighbors = []
            for nb in allneighbors:
                try:
                    if (self.segmentIDs[nb]!=0 and self.segmentIDs[nb]!=segm.label):
                        segmentneighbors.append(self.segmentIDs[nb]) # If adjacent pixel is part of a different subsegment, save it
                except:
                    None
            if len(segmentneighbors) > 0:
                neighbourLabels[p] = segmentneighbors

        return neighbourLabels

    @staticmethod
    def numOfNeighbours(arr, a):
        # For each pixel==True in arr get number of its neighbours==True,
        # given the connectivity matrx a
        arrBinary = np.zeros_like(arr, dtype='uint8')   # Use 'uint8' because rank.sum give wrong values for the boolean arr
        arrBinary[np.where(arr)] = 1
        nneighbours = rank.sum(arrBinary, a)            # sum of neighbouring pixels
        return nneighbours

    #@classmethod
    def findSegmentPixels(self):
        # CREATE DICTIONARY WITH KEY=ID, VALUE=LIST OF SEGMENT PIXELS
        # segmentPixels: for each segment i segmentPixels[i] contains all pixels of segment i
        keys = range(1,np.amax(self.segmentIDs)+1) #TODO: sort(unique(segmentIDs))
        self.segmentPixels = {key: [] for key in keys}
        pixels = zip(*np.where(self.segmentIDs>0))
        for p in pixels:
            l = self.segmentIDs[p]
            self.segmentPixels[l].append(p)


    #@classmethod
    def findSegmentType(self):#TODO: replace segmentIDs with segmentPixels
        # CREATE DICTIONARY WITH KEY=ID, VALUE=SEGMENT TYPE
        # segmentType: Type of each segment; 1 = tip; 2 = branch; 3 = junction
        keys = range(1, np.amax(self.segmentIDs)+1) # TODO: Take keys from segmentPixels
        self.segmentType = {key: None for key in keys}
        pixels = zip(*np.where(self.segmentIDs>-1))
        for p in pixels:
            l = self.segmentIDs[p]
            self.segmentType[l] = self.nneighbours[p]

    #@classmethod
    def createSegments(self):
        keys = range(1,np.amax(self.segmentIDs)+1) #TODO: sort(unique(segmentIDs))
        self.segments = {key: None for key in keys}
        for key in keys:
            pixels = zip(*self.segmentPixels[key])
            self.segments[key] = Segment(key,self.segmentPixels[key],self.segmentType[key],self.distance[pixels])
            if self.segmentType[key] == 2:
                self.segments[key].sort()

    def classifiyTips(self, distToEdge, dist=10.0):
        # Computes all tips that are close to the main root

        for s in self.segments.values():

            distances = distToEdge[zip(*s.pixels)]
            s.minDistToEdge = min(distances[0],distances[-1])
            s.maxDistToEdge = max(distances[0],distances[-1])

            if s.type == 1:
                distToEdge[s.pixels[0]]
                if dist >= distToEdge[s.pixels[0]]:
                    s.isRoot = True
                    s.isTip = False
                else:
                    s.isRoot = False
                    s.isTip = True

    def getSegmentsSize(self):
        sizes = {}
        for s in self.segments:
            sizes[s] = self.segments[s].size()
        return sizes


class BBox:
    def __init__(self,min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def intersects(self, other):
        if self.min_x > other.max_x or self.max_x < other.min_x:
            return False
        if self.min_y > other.max_y or self.max_y < other.min_y:
            return False
        return True

    def union(self,other):
        min_x = min(self.min_x, other.min_x)
        max_x = max(self.max_x, other.max_x)
        min_y = min(self.min_y, other.min_y)
        max_y = max(self.max_y, other.max_y)
        return BBox(min_x, max_x, min_y, max_y)

    def get(self):
        return [self.min_x, self.max_x, self.min_y, self.max_y]

class Segment:
    def __init__(self,label,pixels,segmenType,distance):
        self.label = label
        self.pixels = pixels
        self.type = segmenType
        self.distance = distance
        self.neighbors = {}
        self.isSorted = False
        self.direction = None
        self.isBranch = self.isBranch()
        self.minDistToEdge = None
        self.maxDistToEdge = None
        self.curve = None
        self.segment1 = None
        self.segment2 = None

    # Returns bounding box of pixels
    def bbox(self):
        x,y = zip(*self.pixels)
        #return [min(x),max(x), min(y),max(y)]
        return BBox(min(x),max(x),min(y),max(y))

    # Returns the number of pixels in the segment
    def size(self):
        return len(self.pixels)

    def isBranch(self):
        if self.type == 2:
            return True
        else:
            return False

    # Returns the center of the segment
    def position(self):
        distI = np.inf
        for i, coord in enumerate(self.pixels):
            distJ = -np.inf
            for j, coord2 in enumerate(self.pixels):
                distance = np.linalg.norm(np.array(coord)-np.array(coord2))
                if distJ < distance:
                    distJ = distance
            if distI > distJ:
                distI = distJ
                centerCoord = coord
        return centerCoord


    def setDirection(self):
         # Stop if only one pixel. No need to sort
        if self.type != 2:
            return False

        if len(self.pixels) < 2:
            pixel = self.pixels[0]
            nn = self.neighbors[pixel]
            self.direction = (nn[0],nn[-1])
            return True

        try:
            nStart = self.neighbors[self.pixels[0]]
            nEnd = self.neighbors[self.pixels[-1]]
        except:
            nStart = []
            nEnd = []

        if len(nStart)==1 and len(nEnd)==1:
            self.direction = (nStart[0],nEnd[0])
            return True
        else:
            return False

    # Sorts pixels from one end to the other
    # Works only on branches (tips. i.e. 1 pixel, and junctions cannot be sorted!)
    def sort(self):
        g = nx.Graph()

        # Stop if only one pixel. No need to sort
        if len(self.pixels) < 2:
            return True

        for i in range(len(self.pixels)):
            for j in range(i,len(self.pixels)):
                if (i != j) & (abs(self.pixels[i][0]-self.pixels[j][0]) <= 1) & (abs(self.pixels[i][1]-self.pixels[j][1]) <= 1):
                    g.add_edge(i,j)

        if not nx.is_connected(g):
            raise ValueError('Invalid connectivity in Segment.sort: All pixels must be connected')

        tips = [i for i in g.nodes() if g.degree(i)==1]

        if len(tips) != 2:
            return False
            #raise ValueError('Invalid number of tips in Segment.sort: Segment can only have 2 tips')

        p = nx.shortest_path(g,tips[0],tips[1])

        # Rearrange points along shortest path
        self.pixels = [self.pixels[i] for i in p]
        self.distance = [self.distance[i] for i in p]
        self.isSorted = True
        return True

    def reverse(self):
        self.pixels.reverse()
        self.direction = (self.direction[1],self.direction[0])
        self.distance.reverse()


class Graph:

    def __init__(self):

        self.graph = gt.Graph(directed=False) # Graph of all segments
        self.vertices = {}

    def create(self,segments):
        
        # Property map for segment labels
        self.graph.vertex_properties['label'] = self.graph.new_vertex_property('int')

        # Add node for each segment
        for key in segments.keys():
            v = self.graph.add_vertex()
            self.vertices[key] = v
            self.graph.vertex_properties['label'][v] = key

        
        # For each vertex in graph add edge to neighbor segments
        for v in self.graph.vertices():

            # Get neighboring segments
            v_label = self.graph.vertex_properties['label'][v]
            v_neighbors = segments[v_label].neighbors    

            # For each neighbor add an edge
            for neighbors in v_neighbors.values():
                for nn in neighbors:
                    if nn not in self.vertices:
                        raise ValueError('Node=%g does not exist!' % nn)
                    v_n = self.vertices[nn]
                    if self.graph.edge(v, v_n) is None:
                        self.graph.add_edge(v, v_n) 
        return self.graph