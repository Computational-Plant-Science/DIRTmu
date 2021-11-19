# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 13:41:23 2018

@author: pp34747
"""

import numpy as np
import math


class line:

    __slots__ = ["x","y"]
    
    def __init__(self,x,y):
        # line consisting of x and y components
        self.x = np.array(x, dtype='float32')
        self.y = np.array(y, dtype='float32')

    def xy(self,index):
        return (self.x[index],self.y[index])
        
    def size(self):
        return len(self.x)
    
    def append(self,newline):
        self.x = np.append(self.x, newline.x)
        self.y = np.append(self.x, newline.x)
    
    def merge(self,newline):
        x = np.append(self.x, newline.x)
        y = np.append(self.x, newline.x)
        return line(x,y)
    
    def length(self):
        x_diff = np.diff(self.x)
        y_diff = np.diff(self.y)
        l = np.sum(np.sqrt(x_diff * x_diff + y_diff * y_diff))
        return l
    
    def cumulativelength(self):
        cLength = [0.]
        for i in range(1,len(self.x)):
            cLength.append (cLength[i-1] +
                            math.sqrt((self.x[i]-self.x[i-1])**2 + (self.y[i]-self.y[i-1])**2))
        return cLength
    
    def segmentlengths(self):
        x_diff = np.diff(self.x)
        y_diff = np.diff(self.y)
        l = np.sqrt(x_diff * x_diff + y_diff * y_diff)
        return l
    
    def curvature(self):
        dx = np.gradient(self.x)    # First derivative of x
        ddx = np.gradient(dx)       # Second derivative of x
        dy = np.gradient(self.y)    # First derivative of y
        ddy = np.gradient(dy)       # Second derivative of y
        
        num = dx * ddy - dy * ddx
        denom = (dx * dx + dy * dy)**1.5 
                
        result = num/denom
        result[np.where(denom==0)] = 0.0 #TODO: this can only be result of identical points (where dx=0 and dy=0) => Fix
        return result
    
    def totalcurvature(self):
        # total curvature: 
        # curvature integrated over length
        kappa = self.curvature()
        l = self.segmentlengths()
        tc = np.abs(kappa[:-1] + kappa[1:]) / 2.0
        tc = np.sum(tc * l)
        return tc
    
    def strainenergy(self):
        # total squared energy: 
        # squared curvature integrated over length
        kappa = self.curvature()
        l = self.segmentlengths()
        se = (kappa[:-1] + kappa[1:]) / 2.0
        se = np.sum(se * se * l)
        return se   