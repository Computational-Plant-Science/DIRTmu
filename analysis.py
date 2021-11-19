# -*- coding: utf-8 -*-
"""
This programm post-processes root hairs (removes outliers) and returns results
"""
import pandas as pd

class PostProcess:
    def __init__(self, validity=True, thresh_l2d_ratio=0.0, thresh_connectivity=0.0):
        self.validity = validity
        self.thresh_l2d_ratio = thresh_l2d_ratio
        self.thresh_connectivity = thresh_connectivity
        
    
    def run(self, candidates, get_longest=False, return_outliers=False):
        self.inliers = []
        self.outliers = []
        
        if isinstance(candidates, dict):
            if get_longest:
                temp = []
                for cluster in candidates.values():
                    best = cluster[0]
                    for cand in cluster:
                        if cand.length_total() > best.length_total():
                            best = cand
                    temp.append(best)
            else:
                temp = [item for sublist in candidates.values() for item in sublist] #flatten
            candidates = temp
            
        elif isinstance(candidates, list):
            pass
        else:
            if return_outliers:
                return candidates, []
            else:
                return candidates
            
            
        is_inlier = [True for i in range(len(candidates))]
        
        # For all candidates
        for ind, cand in enumerate(candidates):
            cand.fitCurve()
            # Test for valdidty
            if self.validity:
                is_short = len(cand.path) <= 3 
                has_two_tips = cand.segments[cand.path[0]].type == 1 \
                                    and cand.segments[cand.path[-1]].type == 1
                if is_short and not has_two_tips:
                    is_inlier[ind] = False
            # 
            if cand.length2diameter() < self.thresh_l2d_ratio:
                is_inlier[ind] = False
                
            d_min, d_max = cand.connectivity()
            connectivity = float(d_max-d_min)/d_max
            if connectivity < self.thresh_connectivity:
                is_inlier[ind] = False
                 
        for ind, val in enumerate(is_inlier):
            if val:
                self.inliers.append(candidates[ind])
            else:
                self.outliers.append(candidates[ind])
        
        if return_outliers:
            return self.inliers, self.outliers
        else:
            return self.inliers
       
    
class Results:
    
    def __init__(self, length_mu=True, total_curvature=True, mean_curvature=True,
                 median_curvature=True, min_curvature=True, max_curvature=True,
                 min_diameter=True, max_diameter=True, mean_diameter=True, 
                 median_diameter=True, l2t_ratio=False, 
                 connectivity=False, root_class=True, root_position=True):
        
        self.length_mu = length_mu
        self.total_curvature = total_curvature
        self.l2t_ratio = l2t_ratio
        self.connectivity = connectivity      
        self.mean_curvature = mean_curvature
        self.median_curvature = median_curvature
        self.min_curvature = min_curvature
        self.max_curvature = max_curvature
        self.min_diameter = min_diameter
        self.max_diameter = max_diameter
        self.mean_diameter = mean_diameter
        self.median_diameter = median_diameter
        self.root_class = root_class
        self.root_position = root_position

    
    def get(self, roothairs, pixel_size, root_classes = [], root_positions=[]):
        '''
        Roothairs must be a flattened list of candidate objects
        '''
        data = {}
        
        if self.length_mu:
            data['length_mu'] = [rh.length_total() * pixel_size for rh in roothairs]
            
        if self.total_curvature:
            data['total_curvature'] = [rh.totalcurvature() for rh in roothairs]
            
        if self.mean_curvature:
            data['mean_curvature'] = [rh.mean_curvature() for rh in roothairs]
            
        if self.median_curvature:
            data['median_curvature'] = [rh.median_curvature() for rh in roothairs]
            
        if self.min_curvature:
            data['min_curvature'] = [rh.min_curvature() for rh in roothairs]
            
        if self.max_curvature:
            data['max_curvature'] = [rh.max_curvature() for rh in roothairs]
            
        if self.min_diameter:
            data['min_diameter_mu'] = [rh.min_diameter() * pixel_size for rh in roothairs]
            
        if self.max_diameter:
            data['max_diameter_mu'] = [rh.max_diameter() * pixel_size for rh in roothairs]
            
        if self.mean_diameter:
            data['mean_diameter_mu'] = [rh.mean_diameter() * pixel_size for rh in roothairs]
            
        if self.median_diameter:
            data['median_diameter_mu'] = [rh.median_diameter() * pixel_size for rh in roothairs]
            
        if self.l2t_ratio:
            data['l2t_ratio'] = [rh.length2diameter() for rh in roothairs]
            
        if self.connectivity:
            data['connectivity'] = [rh.connectivity() for rh in roothairs]

        if self.root_class:
            data['root_classes'] = [val for val in root_classes]

        if self.root_position:
            data['root_positions'] = [val for val in root_positions]
            
        # put root hair imformation into tabel
        table = pd.DataFrame(data)
        table.index = table.index + 1
        return table
    
    def out(self, data, roothairs, path_name):
        import numpy as np
        import pickle
        
        curves = []
        for rh in roothairs:
            curves.append(rh.curve)
        
        output = open(path_name, 'wb')
        pickle.dump(curves, output)
        output.close()
            
#        import matplotlib.pyplot as plt
#        
#        uni = np.unique(np.array(data))
#        thresholds = [(a + b) / 2. for a, b in zip(uni[:-1], uni[1:])]
#        
#        my_dpi = 96
#        size = np.array(data.shape)
#        figsize = float(size[1])/my_dpi,float(size[0])/my_dpi
#        fig = plt.figure(figsize=figsize, dpi=my_dpi,frameon=False)
#
#        ax = plt.Axes(fig, [0., 0., 1., 1.])
#        ax.set_axis_off()
#        n_components = len(roothairs)
#        fig.add_axes(ax)
#        plt.contour(data, thresholds, colors='k', linewidths=1., zorder=2)
#        ax = plt.gca()
#        randOrder = np.arange(n_components)
#        np.random.shuffle(randOrder)
#        for counter, rh in enumerate(roothairs):
#            rgba = plt.cm.Spectral(float(np.clip(randOrder[counter], 0, n_components))/n_components)
#            rgb = rgba[0:3]
#            ax.plot(rh.curve.y, rh.curve.x, '-', color=rgb, linewidth=2)
#            #ax.plot(l.y[0],l.x[0],'o', color=rgb, linewidth=3)
#            #ax.plot(l.y[-1],l.x[-1],'o', color=rgb, linewidth=3)
#        ax.set_frame_on(False)
#        ax.set_xticks([]); ax.set_yticks([])
#        plt.axis('equal')
##        plt.show()
##        plt.pause(2.0)
#        plt.savefig(path+name+'_plot.png',dpi=my_dpi)
#        
#        plt.close()