# -*- coding: utf-8 -*-
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


def plot_two_curves(rh1, rh2, data, path):
    fig = Figure(figsize=(10, 10), dpi=50)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(data, cmap='gray')

    ax.plot(rh1[1], rh1[0], color='red', linewidth=1)
    ax.plot(rh2[1], rh2[0], color='blue', linewidth=1)

    ax.set_xlim((0,data.shape[1]))
    ax.set_ylim((data.shape[0],0))

    fig.savefig(path,dpi=50, bbox_inches='tight')

def plot_results(curves, data, path):

    n_components = len(curves)

    fig = Figure(figsize=(10, 10), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(data, cmap='gray')

    randOrder = np.arange(n_components)
    np.random.shuffle(randOrder)
    for counter, rh in enumerate(curves):
        rgba = plt.cm.Spectral(float(np.clip(randOrder[counter], 0, n_components))/n_components)
        rgb = rgba[0:3]
        ax.plot(rh.y, rh.x, color=rgb, linewidth=0.5)

    ax.set_xlim((0,data.shape[1]))
    ax.set_ylim((data.shape[0],0))

    fig.savefig(path,dpi=300, bbox_inches='tight')

def plot_candidates(curves, data, path):

    n_components = len(curves)

    fig = Figure(figsize=(10, 10), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(data, cmap='gray')

    randOrder = np.arange(n_components)
    np.random.shuffle(randOrder)
    for counter, rh in enumerate(curves):
        rgba = plt.cm.Spectral(float(np.clip(randOrder[counter], 0, n_components))/n_components)
        rgb = rgba[0:3]
        ax.plot(rh[1], rh[0], color=rgb)

    ax.set_xlim((0,data.shape[1]))
    ax.set_ylim((data.shape[0],0))

    fig.savefig(path+"candidates_all.png",dpi=100, bbox_inches='tight')


    for counter, rh in enumerate(curves):
        fig = Figure(figsize=(10, 10), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(data, cmap='gray')
        rgba = plt.cm.Spectral(float(np.clip(randOrder[counter], 0, n_components))/n_components)
        rgb = rgba[0:3]
        ax.plot(rh[1], rh[0], color=rgb,  linewidth=2)
        fig.savefig(path+"candidates_"+str(counter)+".png",dpi=100, bbox_inches='tight')


def plot_colored_state(curves, values, min_val, max_val, data, name_path, title=None):

    n_components = len(curves)

    fig = Figure(figsize=(20, 20), dpi=150)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(data, cmap='gray')
    #print im.origin
    randOrder = np.arange(n_components)
    np.random.shuffle(randOrder)
    for counter, rh in enumerate(curves):
        rgba = plt.cm.Spectral(float(np.clip(values[counter], min_val, max_val))/(max_val-min_val))
        rgb = rgba[0:3]
        ax.plot(rh[1], rh[0], color=rgb)

    if title is not None:
        ax.set_title(title) 

    #im.set_extent((0,data.shape[1],data.shape[0],0))
    ax.set_xlim((0,data.shape[1]))
    ax.set_ylim((data.shape[0],0))
    
    fig.savefig(name_path,dpi=150, bbox_inches='tight')

def plot_colored_candidates(curves, values, data, path):

    n_components = len(curves)

    fig = Figure(figsize=(10, 10), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(data, cmap='gray')

    randOrder = np.arange(n_components)
    np.random.shuffle(randOrder)
    for counter, rh in enumerate(curves):
        rgba = plt.cm.Spectral(float(np.clip(values[counter], min(values), max(values)))/(max(values)-min(values)))
        rgb = rgba[0:3]
        ax.plot(rh.y, rh.x, color=rgb)
        
    ax.set_xlim((0,data.shape[1]))
    ax.set_ylim((data.shape[0],0))

    fig.savefig(path+"candidates_all.png",dpi=100, bbox_inches='tight')


    for counter, rh in enumerate(curves):
        fig = Figure(figsize=(10, 10), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(data, cmap='gray')
        rgba = plt.cm.Spectral(float(np.clip(values[counter], min(values), max(values)))/(max(values)-min(values)))
        rgb = rgba[0:3]
        ax.plot(rh.y, rh.x, color=rgb,  linewidth=2)
        ax.set_title('{0:.{1}f}'.format(values[counter], 4)) 
        fig.savefig(path+"candidates_"+str(counter)+".png",dpi=100, bbox_inches='tight')

def sa_plots(candidate_arr, summary, classes):

    import numpy as np

    for j,s in summary.items():
        n_components = len(s['sols'][0])
        randOrder = np.arange(n_components)
        np.random.shuffle(randOrder)
        for i,sol in enumerate(s['sols']):
            if i%5 != 0:
                continue
            sol = np.where(sol)[0]

            fig = plt.figure(figsize=(10,5))
            grid = plt.GridSpec(5, 4, hspace=0.1, wspace=0.5)
            ax1 = fig.add_subplot(grid[0:5, 0:2])
            ax2 = fig.add_subplot(grid[1:4, 2:4])

#            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(classes, cmap='gray')


#            roothairs = [candidate_arr[i_cand] for i_cand in sol]
            for rh_id in sol:
                rh = candidate_arr[rh_id]
                rgba = plt.cm.Spectral(float(np.clip(randOrder[rh_id], 0, n_components))/n_components)
                rgb = rgba[0:3]
                ax1.plot(rh.curve.y, rh.curve.x, '-', color=rgb, linewidth=2)
                #ax.plot(l.y[0],l.x[0],'o', color=rgb, linewidth=3)
                #ax.plot(l.y[-1],l.x[-1],'o', color=rgb, linewidth=3)
#            ax1.set_frame_on(False)
            ax1.set_xticks([]); ax1.set_yticks([])

            T_arr = s['T'][:i+1]
            sub_cost_arr = np.exp(s['sub_cost'][:i+1])
            cost_arr = s['cost'][:i+1]
            best_arr = s['best'][:i+1]
            ax2.plot(range(len(T_arr)), zip(*sub_cost_arr)[0],'r-',linewidth=0.5)
            ax2.plot(range(len(T_arr)), zip(*sub_cost_arr)[1],'b-',linewidth=0.5)
            ax2.plot(range(len(T_arr)), zip(*sub_cost_arr)[2],'g-',linewidth=0.5)
            ax2.plot(range(len(T_arr)), zip(*sub_cost_arr)[3],'orange',linewidth=0.5)
            ax2.plot(range(len(T_arr)), cost_arr,'k-',linewidth=1)
            ax2.plot(range(len(T_arr)), best_arr,'k:',linewidth=1)
            ax2.set_xlim(0,len(s['sols']))
            ax2.set_ylim(1,max(s['cost']))
            ax2.set_xlabel('iteration')
            ax2.set_ylabel('cost')
            ax2.legend(['Strain Energy','Completeness','Min. Distance','Max. Distance','Overall'])

            plt.savefig('C:/Projects/Roothair/Conferences/IPPS 2019/main_for_figures/sa_plots/fig_'+str(j)+'_'+str(i)+'.png',dpi=300)
            plt.close()
        break