# -*- coding: utf-8 -*-
"""
This program creates candidates of root hairs
"""


import numpy as np
import warnings
import networkx as nx
from lines import line
from scipy.interpolate import splprep, splev
from scipy import signal
from scipy import interpolate
from collections import defaultdict
from itertools import product, groupby
from operator import itemgetter
import graph_tool.all as gt
import os
import psutil

def pipeline(graph, segments):
    
    set_weights(graph, segments)
    
    # Create subgraphs for each component of connected roothairs
    comp, hist = gt.label_components(graph)
    candidates = {}
    dummies = {}
    
    print "**************************************************"
    print "             Getting candidates                   "
    print "      # of connected components:" + str(len(hist))   
    print "**************************************************"
    
    pid = os.getpid()
    py = psutil.Process(pid)

    # Run for each subgraph separately for higher efficiency
    for i_comp in range(len(hist)):
        
        print "Component "+str(i_comp)+":  "+str(hist[i_comp])+" nodes"
        
        subgraph = gt.GraphView(graph, vfilt=comp.a == i_comp)

        # Generate possible candidate roothairs
        candgen = CandidateGenerator(max_total_curvature=2.*np.pi)
        dummies[i_comp] = candgen.create_dummies(subgraph, segments)   
        candidates[i_comp] = candgen.create_candidates(subgraph, segments)

        memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
        print " - "+str(len(candidates[i_comp]))+" candidates; memory use: "+ str(round(memoryUse,4))   

    return candidates, dummies

def set_weights(graph, segments):
    # Create internal edge property map for weights
    # Weights are based on length of fitted curve

    # Nodes are tip, branches or branching-points
    # Branches do not act as edges
    #   Edges connect all types of nodes
    #   including tips to branching-points
     
    graph.edge_properties['weight'] = graph.new_edge_property('double')

    # For all edges in graph
    for e in graph.edges():                                     

        v_src = e.source()
        v_trg = e.target()
        
        if v_src.out_degree() == 2:                             # If source vertex is branch segment
            nbrs = list(v_src.all_neighbours())                 
            path = [nbrs[0], v_src, nbrs[1]]     
        elif v_trg.out_degree() == 2:                           # If target vertex is branch segment
            nbrs = list(v_trg.all_neighbours())
            path = [nbrs[0],v_trg, nbrs[1]]
        else:                                                   # Else not part of branch segment
            path = [v_src, v_trg]

        # Map to vertices to segment labels
        path = [graph.vertex_properties['label'][v] for v in path]
            
        candidate = Candidate(path, segments)
        candidate.fitCurve(is_dummy=True)
        
        # Calculate weight
        # If it is edge to/from a branch segment only half 
        # length of branch segments is used as weight
        # Else: Full length of edge if  tip -> branching-point
        #                               tip -> tip
        #                               branching-point -> branching-point
        if v_src.out_degree() == 2 or v_trg.out_degree() == 2: 
            weight = candidate.curve.length()/2.
        else: 
            weight = candidate.curve.length()
        
        # Safe in edge property map
        graph.edge_properties['weight'][e] = weight

class CandidateGenerator:

    def __init__(self,max_total_curvature):

        self.max_total_curvature = max_total_curvature


    def create_dummies(self,graph,segments):

        dummies = []
        done = []

        for source in graph.vertices(): # For all branch segments in graph
            if source.out_degree() == 2: #

                nbrs = list(source.all_neighbours())

                nbrs = [graph.vertex_properties['label'][v] for v in nbrs]
                label_src = graph.vertex_properties['label'][source]
                path = [nbrs[0], label_src, nbrs[1]]
                
                if ((nbrs[0], nbrs[1]) in done or 
                    (nbrs[1], nbrs[0]) in done): # has already been processed
                    continue
                if segments[nbrs[0]].type == 1 and segments[nbrs[1]] == 1: # Is a valid root hair: tip-branch-tip
                    continue
                
                dummies.append(path)

                done.append(tuple([nbrs[0],nbrs[1]]))
        
        return dummies


    def create_candidates(self,graph,segments):

        candidates = []
        n_vertices = graph.num_vertices()
        
        pid = os.getpid()
        py = psutil.Process(pid)
        
        for source in graph.vertices():
            
            # Must start at degree 1 (=tip) or 3 (=junction)
            if source.out_degree() == 2: 
                continue
                       
            dist = gt.shortest_distance(graph, source=source, max_dist=20)

            for target in np.where(np.logical_and(dist.a < graph.num_vertices(), dist.a > 0))[0]:

                # Target node must have degree 1 (=tip) or 3 (=junction)
                if graph.vertex(target).out_degree() == 2: 
                    continue
                
                # Only need to consider one direction of path
                if int(target) >= int(source):
                    continue

                paths = gt.all_paths(graph, source=source, target=target, cutoff=dist.a[target]+2)

                for path in paths:

                    path_labels = [graph.vertex_properties['label'][p] for p in path]
                    candidates.append(path_labels)

        return candidates
        
class Candidate:

    def __init__(self,path,segments):
        self.path = path
        self.segments = segments
        self.pixels, self.diameter, self.pixel_type, self.segment_ids = self.createPixels()
        self.curve = None
        self.is_dummy = None

    def __str__(self):
        return str(self.path)+", "+str(self.curve.length())

    def types(self):
        t = []
        for p in self.path:
            t.append(self.segments[p].type)
        return t
    
    def test_dummy(self):
        if ((len(self.path) == 2 or len(self.path) == 2)
            and (self.segments[self.path[0]].type == 3 or self.segments[self.path[-1]].type == 3)):
            return True
        else:
            return False
        
    def reference_strain(self):       
        indices = [index for index, val in enumerate(self.path) if self.segments[val].type == 1 or self.segments[val].type == 3]
        start_end = zip(indices,indices[1:])
        se = 0.
        for start, end in start_end:
            
            sub_path = [self.path[i] for i in range(start, end+1)]
            sub_candidate = Candidate(sub_path,self.segments)
            sub_candidate.fitCurve(is_dummy=True)
            se = se + sub_candidate.curve.strainenergy()
        return se
    
    def reference_curvature(self):       
        indices = [index for index, val in enumerate(self.path) if self.segments[val].type == 1 or self.segments[val].type == 3]
        start_end = zip(indices,indices[1:])
        tc = 0.
        for start, end in start_end:
            
            sub_path = [self.path[i] for i in range(start, end+1)]
            sub_candidate = Candidate(sub_path,self.segments)
            sub_candidate.fitCurve(is_dummy=True)
            tc = tc + sub_candidate.curve.totalcurvature()
        return tc
        
    # Concatanate individual branches in order of path
    def createPixels(self):
        pixels = []
        diameter = []
        pixel_type = []
        #segment_id = []
        lengths = []
        for i,p in enumerate(self.path):
            segment = self.segments[p]
            if segment.type != 2: # is tip or junction
                pixels.append(np.mean(segment.pixels,0).tolist())
                diameter.append(2.*np.mean(segment.distance))
                pixel_type.append(segment.type)
                #segment_id.append(segment.label)
                lengths.append(1)
            elif segment.size() == 1:
                pixels.append(tuple(segment.pixels[0]))
                diameter.append(2.*segment.distance[0])
                pixel_type.append(segment.type)
                #segment_id.append(segment.label)
                lengths.append(1)
            else:

                direction_ok = True
                if i+1 < len(self.path):
                    next_p = self.path[i+1]
                    if segment.direction[1] != next_p:
                        direction_ok = False
                if i > 0:
                    prev_p = self.path[i-1]
                    if segment.direction[0] != prev_p:
                        direction_ok = False

                # reverse
                if not direction_ok:
                    newpoints = list(reversed(segment.pixels))
                    newdiam = list(reversed(segment.distance))
                else:
                    newpoints = segment.pixels
                    newdiam = segment.distance
                for point in newpoints:
                    pixels.append(point)
                    pixel_type.append(segment.type)
                    #segment_id.append(segment.label)
                for d in newdiam:
                    diameter.append(2.*d)
                
                lengths.append(segment.size())

        segment_id = []
        start_id = 0
        for n in lengths:
            end_id = start_id + n
            segment_id.append((start_id, end_id))
            start_id = end_id

        return pixels, diameter, pixel_type, segment_id

    def median_diameter(self):
        return np.median(self.diameter)

    def min_diameter(self):
        return min(self.diameter)
    
    def max_diameter(self):
        return max(self.diameter)
    
    def mean_diameter(self):
        return np.mean(self.diameter)
    
    def length(self):
        if self.curve is None:
            warnings.warn("Curve does not exist. Curve length is estimated from number of pixels.")
            return len(self.pixels)
        elif self.curve.size() == 0:
            warnings.warn("No points in curve. Curve length set to 0.")
            return 0.0
        else:
            return self.curve.length()
    
    def length_total(self):
        tip_length = 0.
        if len(self.diameter) == 1 and self.pixel_type[0]==1:
            tip_length += self.diameter[0]/2.0
        else:
            if self.pixel_type[0]==1:
                tip_length += self.diameter[0]/2.0 
            if self.pixel_type[-1]==1:
                self.diameter[-1]/2.0

        return self.length() + tip_length
        
    def strainenergy(self):
        if self.curve is None:
            warnings.warn("Curve does not exist. Strain energy set to 0.")
            return 0.0
        elif self.curve.size() == 0:
            warnings.warn("No points in curve. Strain energy set to 0.")
            return 0.0
        else:
            return self.curve.strainenergy()

    def totalcurvature(self):
        if self.curve is None:
            warnings.warn("Curve does not exist. Total curvature set to 0.")
            return 0.0
        elif self.curve.size() == 0:
            warnings.warn("No points in curve. Total curvature set to 0.")
            return 0.0
        else:
            return self.curve.totalcurvature()
    
    def mean_curvature(self):
        if self.length() > 0:
            return self.totalcurvature()/self.length()
        else:
            return 0.0
        
    def median_curvature(self):
        return np.median(abs(self.curve.curvature()))

    def min_curvature(self):
        return min(abs(self.curve.curvature()))
    
    def max_curvature(self):
        return max(abs(self.curve.curvature()))
    
    def connectivity(self):
        d1 = self.segments[self.path[0]].minDistToEdge
        d2 = self.segments[self.path[-1]].minDistToEdge
        return min(d1,d2), max(d1,d2)
        
    def length2diameter(self):
        try:
            return self.length_total()/self.median_diameter()
        except ZeroDivisionError:
            warnings.warn("Length is 0.")
            return 0.0

    def fitCurve(self, is_dummy=False):
        x, y = np.array(zip(*self.pixels))
        if is_dummy:
            w = np.ones_like(self.diameter,dtype=float)
        else:
            
            x_diam = range(len(self.diameter))
            y_diam = self.diameter
            
            window_length = min(len(y_diam), 51)
            if window_length % 2 == 0:
                window_length = window_length - 1
            polyorder = min(window_length-1,3)
            y_diam_hat = signal.savgol_filter(y_diam, window_length, polyorder)

            peaks, props = signal.find_peaks(-y_diam_hat)
            peaks = list(peaks)
                        
            if len(peaks) > 0:
                if not peaks[0] == 0:
                    peaks.insert(0, 0)
                if not peaks[-1] == x_diam[-1]:
                    peaks.append(x_diam[-1])
            else:
                peaks = [x_diam[0], x_diam[-1]]
                
            f = interpolate.interp1d(peaks, [y_diam_hat[p] for p in peaks])
            y_diam_new = f(x_diam)
            
            y_diam_adjusted = np.array(y_diam)-np.array(y_diam_new)
            y_diam_adjusted = [max(0.25, 1.0+v) for v in y_diam_adjusted]
            
            w = 1./np.array(y_diam_adjusted)
        
        s = len(w)
        w[0] = 1000
        w[-1] = 1000
        
        try:
            tck, u = splprep([x,y],w=w,s=s)
            new_points = splev(u, tck)
            #new_points[0][0] = x[0]
            #new_points[0][-1] = x[-1]
            #new_points[1][0] = y[0]
            #new_points[1][-1] = y[-1]
            self.curve = line(new_points[0],new_points[1])
        except:
            self.curve = line(x,y)
        return True

class ReferenceValues:
    def __init__(self,measure='curvature', use_ref_tips=False):
        self.min = {}
        self.max = {}
        self.use_ref_tips = use_ref_tips
        if measure=='curvature':
            self.measure = 1
        elif measure=='strain':
            self.measure = 2
        else:
            print " Warning: "+measure+" is invalid type.\nmeasure=curvature will be used instead."
            self.measure = 1

    def add(self, candidate):
        kappa = candidate.curve.curvature()
        l = candidate.curve.segmentlengths()
        tc = np.abs(kappa[:-1] + kappa[1:]) / 2.0

        if self.measure == 1:
            value_list = tc * l
        else:
            value_list = tc * tc * l
        
        size_path = len(candidate.path) 
        for i,p in enumerate(candidate.path):
            segment = candidate.segments[p]

            if segment.type == 2:

                sub_path = [i-1,i]
                ids = np.array([ind for i in sub_path for ind in range(candidate.segment_ids[i][0], candidate.segment_ids[i][1])])

                value_sum = sum(value_list[ids])
                

                if self.use_ref_tips:
                    identifier = [p]

                    if i-1 == 0: # if previous segment is first segment
                        pre = candidate.path[i-1]
                    else:
                        pre = None

                    if i+1 == size_path-1: # if next segment is last segment
                        post = candidate.path[i+1]
                    else:
                        post = None

                    if pre is not None and post is not None:
                        identifier.append(min(pre, post))
                        identifier.append(max(pre, post))
                    elif pre is not None:
                        identifier.append(pre)
                    elif post is not None:
                        identifier.append(post)
                    
                    identifier = tuple(identifier)
                else:
                    identifier = p

                if identifier in self.min:
                    self.min[identifier] = min(value_sum, self.min[identifier])
                else:
                    self.min[identifier] = value_sum
                
                if identifier in self.max:
                    self.max[identifier] = max(value_sum, self.max[identifier])
                else:
                    self.max[identifier] = value_sum

                    
    def calc(self, path, segments, segment_ids):

        min_value = 0.0
        max_value = 0.0

        size_path = len(path) 
        for i,p in enumerate(path):
            segment = segments[p]

            if segment.type == 2:

                # sub_path = [i-1,i]
                # ids = np.array([ind for i in sub_path for ind in range(segment_ids[i][0], segment_ids[i][1])])

                if self.use_ref_tips:
                    identifier = [p]

                    if i-1 == 0: # if previous segment is first segment
                        pre = path[i-1]
                    else:
                        pre = None

                    if i+1 == size_path-1: # if next segment is last segment
                        post = path[i+1]
                    else:
                        post = None

                    if pre is not None and post is not None:
                        identifier.append(min(pre, post))
                        identifier.append(max(pre, post))
                    elif pre is not None:
                        identifier.append(pre)
                    elif post is not None:
                        identifier.append(post)
                    
                    identifier = tuple(identifier)
                else:
                    identifier = p

                min_value += self.min[identifier]
                max_value += self.max[identifier]
        
        return min_value, max_value

    
class Conflicts:
    def __init__(self, paths, lines, segment_ids, segments, data):
        self.paths = paths
        self.lines = lines
        self.segment_ids = segment_ids
        self.segments = segments
        self.data = data

    def create(self):
        # Returns adjecency list with conflicts between canditate intersection lines

        conf_list = [[] for i in self.paths]    # List of candidates that are conflicting
        merge_list = [[] for i in self.paths]   # List of candidates that can be merged

        segs = self.candidates_in_segment(self.paths)
        adj_list = self.overlapping_candidates(segs, self.paths) # Adjacency list

        for id_1, id_2_list in adj_list.items():
            for id_2 in id_2_list:
                path_1 = self.paths[id_1]
                path_2 = self.paths[id_2]
                line_1 = self.lines[id_1]
                line_2 = self.lines[id_2]
                segment_ids_1 = self.segment_ids[id_1]
                segment_ids_2 = self.segment_ids[id_2]

                if self.isAdjacent(path_1, path_2, line_1, line_2, segment_ids_1, segment_ids_2):
                    merge_list[id_1].append(id_2)
                    merge_list[id_2].append(id_1)
                    #rh_plot.plot_two_curves(line_1, line_2, self.data, '/mnt/c/Projects/Roothair/temp/result/adjacent/'+str(id_1)+'_'+str(id_2)+'.png')
                elif self.hasConflict(path_1, path_2, line_1, line_2, segment_ids_1, segment_ids_2):
                    conf_list[id_1].append(id_2)
                    conf_list[id_2].append(id_1)
                    #rh_plot.plot_two_curves(line_1, line_2, self.data, '/mnt/c/Projects/Roothair/temp/result/conflicts/'+str(id_1)+'_'+str(id_2)+'.png')
                #else:
                    #rh_plot.plot_two_curves(line_1, line_2, self.data, '/mnt/c/Projects/Roothair/temp/result/no_conflicts/'+str(id_1)+'_'+str(id_2)+'.png')


        return [np.array(a, dtype=int) for a in conf_list], merge_list, adj_list


    def candidates_in_segment(self, candidate_paths):
        """
        Creates a dictionary with keys to segment ids.
        Value is a list with candidate ids that go 
        through segment.
        """

        seg = defaultdict(list)
        for counter, path in enumerate(candidate_paths):
            for p in path:
                seg[p].append(counter)
        return seg

    def overlapping_candidates(self, seg, candidate_paths):
        """
        Creates a list of candidates that overlap in 
        one or more segments.
        """
        overlaps = {}
        for id_1, path in enumerate(candidate_paths):
            overlaps[id_1] = set([id_2 for p in path for id_2 in seg[p] if id_2 > id_1])

        return overlaps

            

    def index(self, subseq, seq):
        '''
        https://stackoverflow.com/questions/425604/best-way-to-determine-if-a-sequence-is-in-another-sequence-in-python
        '''
        """Return an index of `subseq`uence in the `seq`uence.
    
        Or `-1` if `subseq` is not a subsequence of the `seq`.
    
        The time complexity of the algorithm is O(n*m), where
    
            n, m = len(seq), len(subseq)
    
        >>> index([1,2], range(5))
        1
        >>> index(range(1, 6), range(5))
        -1
        >>> index(range(5), range(5))
        0
        >>> index([1,2], [0, 1, 0, 1, 2])
        3
        """
        i, n, m = -1, len(seq), len(subseq)
        try:
            while True:
                i = seq.index(subseq[0], i + 1, n - m + 1)
                if subseq == seq[i:i + m]:
                   return i
        except ValueError:
            return -1
    
    def rule_1(self, path_1, path_2):
        # Rule 1:
        # Starting/ending directions must be different
        # i.e. branching is not allowed
        # Unless close to root surface

        startDir_1 = tuple([path_1[0],path_1[1]])    
        endDir_1 = tuple([path_1[-1],path_1[-2]])

        startDir_2 = tuple([path_2[0],path_2[1]])    
        endDir_2 = tuple([path_2[-1],path_2[-2]])

        s1 = set([startDir_1, endDir_1])
        s2 = set([startDir_2, endDir_2])

        overlap = s1.intersection(s2)
        if len(overlap) > 0:
            for i in overlap:
                if self.segments[i[0]].minDistToEdge > 10:
                    return True
        return False
        
    def rule_2(self, path_1, path_2):
        # Rule 2:
        # Start/end nodes of a path must not overlapp with inner nodes of other path
        # For path 1

        intersect = set(path_1).intersection(path_2)
        
        if path_1[0] in intersect and path_1[1] in intersect:
            return True
        elif path_1[-1] in intersect and path_1[-2] in intersect:
            return True
        elif path_2[0] in intersect and path_2[1] in intersect:
            return True
        elif path_2[-1] in intersect and path_2[-2] in intersect:
            return True
        else:
            return False
    
    def consecutive_groups(self, iterable):
        """
        https://more-itertools.readthedocs.io/en/latest/_modules/more_itertools/more.html#consecutive_groups
        returns groups of consecutive values in iterable
        """
        for k, g in groupby(enumerate(iterable), lambda ix : ix[0] - ix[1]):
            yield map(itemgetter(1), g)

    def intersections(self, list_1, list_2):
        """
        returns all individual intersections of list_1 with list_2
        """
        set_2 = set(list_2)

        # sorted list of indices from list_1 that intersect with list_2
        intersection_ids = [i for i,val in enumerate(list_1) if val in set_2]

        np_list = np.array(list_1)

        for group in self.consecutive_groups(intersection_ids):
            yield np_list[np.array(group)]

    def rule_3(self, path_1, path_2, line_1, line_2, segment_ids_1, segment_ids_2, max_dist_squared=2.0, max_length=50):
        # Rule 3:
        # If overlapping too much with other curve
       
        for intersect in self.intersections(path_1, path_2):

            len_intersect = len(intersect)
            
            if len_intersect > 2:

                intersect = set(intersect)

                sub_path = [i for i,p in enumerate(path_1) if p in intersect]
                ids_1 = np.array([ind for i in sub_path for ind in range(segment_ids_1[i][0], segment_ids_1[i][1])])

                sub_path = [i for i,p in enumerate(path_2) if p in intersect]
                ids_2 = np.array([ind for i in sub_path for ind in range(segment_ids_2[i][0], segment_ids_2[i][1])])

                x_1 = line_1[0][ids_1]
                y_1 = line_1[1][ids_1]

                x_2 = line_2[0][ids_2]
                y_2 = line_2[1][ids_2]
                
                # Determine direction of intersect
                ref_dir = [p for p in path_1 if p in intersect]
                if self.index(ref_dir, path_2) == -1:
                    x_2 = x_2[::-1]
                    y_2 = y_2[::-1]

                x_diff = x_1 - x_2
                y_diff = y_1 - y_2

                dist_sqared = np.add(x_diff*x_diff, y_diff*y_diff)

                if np.count_nonzero(dist_sqared < max_dist_squared) > max_length:
                    return True
                """
                if len_intersect > 9:
                    rh_plot.plot_two_curves(line_1, line_2, self.data, '/mnt/c/Projects/Roothair/temp/result/conflcits/n_nodes_'+str(path_1[0])+'_'+str(path_1[-1])+'_'+str(path_2[0])+'_'+str(path_2[-1])+'_'+str(np.random.randint(100))+'.png')
                if len(dist_sqared) > 300:
                    rh_plot.plot_two_curves(line_1, line_2, self.data, '/mnt/c/Projects/Roothair/temp/result/conflcits/len_'+str(path_1[0])+'_'+str(path_1[-1])+'_'+str(path_2[0])+'_'+str(path_2[-1])+'_'+str(np.random.randint(100))+'.png')
                """
        return False

    def hasConflict(self, path_1, path_2, line_1, line_2, segment_ids_1, segment_ids_2):

        if self.rule_1(path_1, path_2):
            return True

        if self.rule_2(path_1, path_2):
            return True
        
        if self.rule_3(path_1, path_2, line_1, line_2, segment_ids_1, segment_ids_2):
            return True

        return False
    
    def isAdjacent(self, path_1, path_2, line_1, line_2, segment_ids_1, segment_ids_2):

        if len(set(path_1).intersection(set(path_2))) != 3:
            return False
            
        startDir_1 = tuple(path_1[0:3:1])    
        endDir_1 = tuple(path_1[-1:-4:-1])

        startDir_2 = tuple(path_2[2::-1])
        endDir_2 = tuple(path_2[-3::])

        s1 = set([startDir_1, endDir_1])
        s2 = set([startDir_2, endDir_2])

        if len(s1.intersection(s2)) == 1:
            if len(set(path_1).intersection(set(path_2))) == 3 and len(path_1) > 3 and len(path_2) > 3:
                if self.rule_3(path_1, path_2, line_1, line_2, segment_ids_1, segment_ids_2, max_length=5):
                    return True

        return False


class DummyConflicts(Conflicts):
    def __init__(self, paths, paths_dummy):
        self.paths = paths
        self.paths_dummy = paths_dummy

    def create(self):
        
        adj_list = [[] for i in self.paths]

        segs = self.candidates_in_segment(self.paths)
        segs_dummy = self.candidates_in_segment(self.paths_dummy)
        overlaps = self.overlapping_with_dummies(segs, segs_dummy)

        for pair in overlaps:
            if self.hasConflict(self.paths[pair[0]], self.paths_dummy[pair[1]]):
                adj_list[pair[0]].append(pair[1])

        return [np.array(a, dtype=int) for a in adj_list]
    
    def overlapping_with_dummies(self, segs_1, segs_2):
        overlaps = set()
        for key in segs_1:
            overlaps.update(product(segs_1[key],segs_2[key]))
        return list(overlaps)

    def hasConflict(self, path, path_dummy):
        intersect = set(path).intersection(path_dummy)
        if len(intersect) == len(path_dummy):
            return True
        return False