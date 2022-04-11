#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
----------------------------------------------------------------------------------------------------
HairyRoots 1.0 - An automatic high throughput root hair phenotyping algorithm

The software is written in:
- python 2.7 (https://www.python.org)

The software depends on:
- the numpy package (http://sourceforge.net/projects/numpy/)
- the scipy package (http://www.scipy.org/SciPy)
- the graph_tool package
- the scikit image package
----------------------------------------------------------------------------------------------------
Author: Peter Pietrzyk
Department of Plant Biology
University of Georgia
Mail: peterjacek.pietrzyk@uga.edu
Web: http://www.computational-plant-science.org
----------------------------------------------------------------------------------------------------
'''


import argparse
import time
import os
import psutil
import csv

import numpy as np
from PIL import Image
import pandas as pd

import rh_io
import preprocessing
import segmentation
import candidates
import optimization
import analysis
import rh_plot
import rh_density
import graphs


def run_pipeline(args):
    
    time_start = time.time()
    time_intermediate = time_start

    print(args)
    
    meta_data = dict()
    
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use:', round(memoryUse,4))
    elapsed_time = time.time() - time_intermediate
    print('Elapsed time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    meta_data['time_0'] = elapsed_time
    
    # 1. Load data
    time_intermediate = time.time()
    
    experiment_name = os.path.splitext(os.path.basename(args.input_path))[0]
    data = rh_io.load(args.input_path)
    
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use:', round(memoryUse,4))
    elapsed_time = time.time() - time_intermediate
    print('Elapsed time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    meta_data['time_load'] = elapsed_time
    
    # 2. Prepocess and put into correct format
    time_intermediate = time.time()
    prep = preprocessing.Preprocessing(max_distance=args.max_dist, # max distance in pixels
                                       id_root=args.id_root,
                                       id_background=args.id_background,
                                       id_roothair=args.id_roothair,
                                       is_prune=args.is_prune,
                                       is_close_gaps=args.is_close_gaps)

    ma, ma_dist, dist_to_root, data = prep.run(data)

    area_roothair = np.count_nonzero(data==args.id_roothair)
    area_root = np.count_nonzero(data==args.id_root)
    area_background = np.count_nonzero(data==args.id_background)

    if args.print_all is not None:
        prep.out(args.print_all)

    del prep

    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use:', round(memoryUse,4))
    elapsed_time = time.time() - time_intermediate
    print('Elapsed time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    meta_data['time_preprocess'] = elapsed_time
    
    # 3. CREATE SEGMENTS OF MEDIAL AXIS
    time_intermediate = time.time()
    rh_segm = segmentation.Segmentation(ma,ma_dist)
    rh_segm.classifiyTips(dist_to_root, args.thresh_dist_to_root)       # Classifiy tips into root or tip

    # Initialize object for branching graph.
    # Create nodes in graph from dict of segments in medial axis.
    # Connect nodes. Results in all neighborhood information of medial axis
    g = segmentation.Graph()
    graph = g.create(rh_segm.segments)
    vertices = g.vertices

    # TODO:QUICK FIX FOR WRONG SEGMENTTYPES IN bg.segmentType
    for s in rh_segm.segments.values():
        v = vertices[s.label]
        if v.out_degree() == 1:
            s.type = 1
    for v in graph.vertices():
        if v.out_degree() == 1:
            rh_segm.segmentType[graph.vertex_properties['label'][v]] = 1
    
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use:', round(memoryUse,4))
    elapsed_time = time.time() - time_intermediate
    print('Elapsed time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    meta_data['time_segments'] = elapsed_time
    meta_data['n_tips'] = len(np.where(np.array(list(rh_segm.segmentType.values()))==1)[0])
    meta_data['n_junctions'] = len(np.where(np.array(list(rh_segm.segmentType.values()))>2)[0])
    
    # 4. Get candidates
    time_intermediate = time.time()
    rh_candidates, dummies = candidates.pipeline(graph=graph, segments=rh_segm.segments)

    all_candidates = [item for sublist in rh_candidates.values() for item in sublist] #flatten
    all_dummies = [item for sublist in dummies.values() for item in sublist] #flatten

    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use:', round(memoryUse,4))
    meta_data['n_clusters'] = len(rh_candidates)
    meta_data['n_candidates'] = len(all_candidates)
    
    # 4.1 Gather data for dummies
    print('Gathering data for dummies...')
    dummy_lengths = []
    dummy_min_dist = []
    dummy_max_dist = []
    for path in all_dummies:
        d = candidates.Candidate(path, rh_segm.segments)
        d.fitCurve()
        d_min, d_max = d.connectivity()
        dummy_lengths.append(d.length())
        dummy_min_dist.append(d_min)
        dummy_max_dist.append(d_max)
    # Dummy values
    dummy_lengths = np.array(dummy_lengths)
    dummy_min_dist = np.array(dummy_min_dist)
    dummy_max_dist = np.array(dummy_lengths)

    if sum(dummy_min_dist<2) > 0: # Use dummies attached to root; take mean of their max distance to root  
        normDistanceHigh = np.mean(dummy_max_dist[dummy_min_dist<2])
    else:                         # If no dummies are attached to root, use mean of dummy lengths
        normDistanceHigh = np.mean(dummy_lengths)

    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use:', round(memoryUse,4))


    # 4.2 Gather data for candidates

    n_candidates = len(all_candidates)
    print('Gathering data for candidates...')
    print(" - "+str(n_candidates)+" candidate(s)")

    # Reference curvature and strain per segment
    # Does not store candidate inform
    if args.measure == 'strain_energy': 
        ref_segment_strain = candidates.ReferenceValues('strain', args.use_ref_tips)
    ref_segment_curvature = candidates.ReferenceValues('curvature', args.use_ref_tips)

    segment_ids = []
    total_curvature = []

    for i,path in enumerate(all_candidates):
        
        if i%10000 == 0:
            print(' - Candidate '+str(i))

        c = candidates.Candidate(path, rh_segm.segments)
        c.fitCurve(is_dummy=False)
        if args.measure == 'strain_energy': 
            ref_segment_strain.add(c)
        ref_segment_curvature.add(c)
        segment_ids.append(c.segment_ids)               # Append start-end index of segments in curve
        total_curvature.append(c.totalcurvature())      # Append total curvature of curve

    # Calculate reference curvature for each candidate and extract 
    # only candidates with excess curvature < 90 degrees
    min_reference_curvature = []
    for i,path in enumerate(all_candidates):
        min_ref_value,_ = ref_segment_curvature.calc(path, rh_segm.segments, segment_ids[i])
        min_reference_curvature.append(min_ref_value)
    candidate_filter = np.where( np.array(total_curvature) - np.array(min_reference_curvature) <= 0.5 * np.pi )[0]
    good_candidates = [all_candidates[i] for i in candidate_filter]
    print('Keeping '+str(len(good_candidates))+' of '+str(n_candidates)+' candidates')

    # Get information for good candidates
    lines = []
    segment_ids = []
    curve_measure = []
    min_distance = []
    max_distance = []
    min_reference_curve = []

    for i,path in enumerate(good_candidates):

        if i%10000 == 0:
            print(' - Candidate '+str(i))

        c = candidates.Candidate(path, rh_segm.segments)
        c.fitCurve(is_dummy=False)
        
        d_min, d_max = c.connectivity()
        lines.append(np.vstack((c.curve.x, c.curve.y))) # Append curve
        segment_ids.append(c.segment_ids)               # Append start-end index of segments in curve
        if args.measure == 'strain_energy':
            curve_measure.append(c.strainenergy())          # Append strain energy of curve
            min_ref_value,_ = ref_segment_strain.calc(path, rh_segm.segments, c.segment_ids)
        else:
            curve_measure.append(c.totalcurvature())          # Append strain energy of curve
            min_ref_value,_ = ref_segment_curvature.calc(path, rh_segm.segments, c.segment_ids)
        min_reference_curve.append(min_ref_value)
        min_distance.append(d_min)                      # Append min distance to root
        max_distance.append(d_max)                      # Append max distance to root     

    curve_measure = np.array(curve_measure)
    min_distance = np.array(min_distance)
    max_distance = np.array(max_distance)
    min_reference_curve = np.array(min_reference_curve)

    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use:', round(memoryUse,4))

    # Create object to hold all candidate/dummy information to calculate cost
    cand_info = optimization.CandidateInformation()
    cand_info.paths = good_candidates
    cand_info.n_segments = np.array([(len(p)-1)/2 for p in good_candidates])

    # Set information from dummies
    cand_info.dummy_lengths = dummy_lengths

    # Set information from candidates
    cand_info.excess_strain = curve_measure-min_reference_curve
    cand_info.min_distance = min_distance
    cand_info.max_distance = max_distance

    # Minimum distance of segments to root
    cand_info.minDistToEdge = {key: val.minDistToEdge for key, val in rh_segm.segments.items()}
    
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use:', round(memoryUse,4))

    # 4.3 Conflicts: For each roothair get a list of conflicting roothairs
    print(" - Computing conflicts...")
    conflicts = candidates.Conflicts(good_candidates, lines, segment_ids, rh_segm.segments, data)
    conflicts_list, merge_list , adj_list = conflicts.create()

    # 4.4 Compute offset for merged candidates and add merged curves to best curvatures per segment (ref_segment_strain or ref_segment_curvature)
    offset_dict = {}

    for cand_i in range(len(merge_list)):

        if cand_i%1000 == 0:
                print(' - Merging candidate '+str(cand_i))

        for cand_j in merge_list[cand_i]:
            if cand_i>=cand_j:
                continue
            
            # Construct path of connected candidates
            g = graphs.Candidate_Graph(cand_info.paths[cand_i])
            g.merge(graphs.Candidate_Graph(cand_info.paths[cand_j]))
            merged_path = g.get_path()
            c = candidates.Candidate(merged_path, rh_segm.segments)
            c.fitCurve(is_dummy=False)

            if args.measure == 'strain_energy':
                merged_curve_measure = c.strainenergy()                # Calculate strain energy of curve
                ref_segment_strain.add(c)
            else:
                merged_curve_measure = c.totalcurvature()              # Calculate total curvature of curve
                ref_segment_curvature.add(c)

            curvature_offset =  merged_curve_measure - curve_measure[cand_i] - curve_measure[cand_j]
            offset_dict[(cand_i,cand_j)] = curvature_offset

    # Need to recalculate reference curvature for good candidates and for merged candidates
    # Recaluclate reference curvature for good candidates
    min_reference_curve = []
    for i,path in enumerate(good_candidates):
        if i%10000 == 0:
            print(' - Candidate '+str(i))
        c = candidates.Candidate(path, rh_segm.segments)
        if args.measure == 'strain_energy':
            min_ref_value,_ = ref_segment_strain.calc(path, rh_segm.segments, c.segment_ids)
        else:
            min_ref_value,_ = ref_segment_curvature.calc(path, rh_segm.segments, c.segment_ids)
        min_reference_curve.append(min_ref_value)
    min_reference_curve = np.array(min_reference_curve)
    cand_info.min_reference_strain = min_reference_curve

    # Recaluclate information for merged candidates
    for cand_i in range(len(merge_list)):
        if cand_i%1000 == 0:
                print(' - Merging candidate '+str(cand_i))
        for cand_j in merge_list[cand_i]:
            if cand_i>=cand_j:
                continue
            
            # Construct path of connected candidates
            g = graphs.Candidate_Graph(cand_info.paths[cand_i])
            g.merge(graphs.Candidate_Graph(cand_info.paths[cand_j]))
            merged_path = g.get_path()
            c = candidates.Candidate(merged_path, rh_segm.segments)            

            if args.measure == 'strain_energy':
                min_ref_value,_ = ref_segment_strain.calc(merged_path, rh_segm.segments, c.segment_ids)
            else:
                min_ref_value,_ = ref_segment_curvature.calc(merged_path, rh_segm.segments, c.segment_ids)
            ref_value_offset = min_reference_curve[cand_i] + min_reference_curve[cand_j] - min_ref_value
            offset_dict[(cand_i,cand_j)] += ref_value_offset

    """
    curves = []
    for i,path in enumerate(good_candidates):
        c = candidates.Candidate(path, rh_segm.segments)
        c.fitCurve(is_dummy=False)
        curves.append(c.curve)
    """
    #rh_plot.plot_colored_candidates(curves, np.array(curve_measure) - np.array(min_reference_curve), data, "/mnt/c/Projects/Roothair/Images/temp/") 
    #rh_plot.plot_colored_candidates(curves_array, excess_curvature_array, data, "/mnt/c/Projects/Roothair/Images/temp/merged_") 
    

    # For each roothair get a list of conflicting dummies 
    print(" - Computing conflicts with dummies...")
    rh_dummy_conflicts = candidates.DummyConflicts(good_candidates, all_dummies)
    rh_dummy_conflicts_list = rh_dummy_conflicts.create()

    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use:', round(memoryUse,4))

    n_candidates = len(good_candidates)   # number of candidates
    print(" - "+str(n_candidates)+" candidate(s)")

    elapsed_time = time.time() - time_intermediate
    print('Elapsed time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    meta_data['time_candidates'] = elapsed_time
    
    # 5. Optimize
    time_intermediate = time.time()

    # Set weights for optimization and initilize cost object
    weights = [args.w_curve, args.w_len, args.w_mind]
    costCalculator = optimization.Cost(measure=args.measure , cost_type=args.cost_type, weights=weights)
    optimizer = optimization.Optimize(cost=costCalculator, nIterations=args.n_levels) 
    
    # Run optimization
    roothair_paths, solution_summary, sa_parameters = optimizer.run(cand_info, conflicts_list, merge_list, adj_list, rh_dummy_conflicts_list, offset_dict)
    solution_roothairs = [candidates.Candidate(path, rh_segm.segments) for path in roothair_paths]

    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use:', round(memoryUse,4))
    elapsed_time = time.time() - time_intermediate
    print('Elapsed time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    meta_data['time_optimize'] = elapsed_time
    
    # 6. Post Processing
    time_intermediate = time.time()
    post = analysis.PostProcess(validity=False, thresh_l2d_ratio=1.2, thresh_connectivity=0.8) # postprocessing and analysis
    inliers, outliers = post.run(solution_roothairs, get_longest=False, return_outliers=True)
    elapsed_time = time.time() - time_intermediate
    meta_data['time_post'] = elapsed_time


    # **************
    # 7. Analysis
    # **************
    time_intermediate = time.time()

    # Compute root hair density and root class
    summary, root_classes, root_positions, edge_info = rh_density.computeDensity(data, [rh.curve for rh in inliers], rootIdx=args.id_root, pixel_size=args.pixel_size)

    # Root hair measurements
    results = analysis.Results()
    table_roothairs = results.get(inliers, args.pixel_size, root_classes=root_classes, root_positions=root_positions)
    table_outliers = results.get(outliers, args.pixel_size, root_classes=['None']*len(outliers), root_positions=['None']*len(outliers))
    
    summary['Area Roothair (pxl)'] = area_roothair
    summary['Area Root (pxl)'] = area_root
    summary['Area Background (pxl)'] = area_background

    summary['Ratio Completeness'] = solution_summary['SA_resultRatioComplete']
    summary['Length Inliers (mu)'] = sum([c.length() for c in inliers]) * args.pixel_size
    summary['Length Outliers (mu)'] = sum([c.length() for c in outliers]) * args.pixel_size


    elapsed_time = time.time() - time_intermediate
    meta_data['time_analysis'] = elapsed_time


    # *****************
    # 8. Save result
    # *****************
    time_intermediate = time.time()

    rh_io.save_table(table_roothairs, os.path.join(args.output_path, experiment_name+'_roothairs.csv'))
    col_order = sorted(summary.keys(), key=lambda x:x.lower())
    table = pd.DataFrame(summary, index=[experiment_name])
    table = table[col_order]
    rh_io.save_table(table, os.path.join(args.output_path, experiment_name+'_summary.csv'))

    if args.print_all:
        # results.out(data, inliers, os.path.join(args.output_path, experiment_name+'_roothairs.pkl'))
        # results.out(data, outliers, os.path.join(args.output_path, experiment_name+'_outliers.pkl'))
        im = Image.fromarray(data)
        im.save(os.path.join(args.output_path, experiment_name+"_classes.tiff"),compression='tiff_lzw')
        rh_io.save_table(table_outliers,  os.path.join(args.output_path, experiment_name+'_outliers.csv'))

    elapsed_time = time.time() - time_intermediate
    meta_data['time_save'] = elapsed_time
    print('Elapsed time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


    # ******************
    # 9. Plot results
    # ******************
    time_intermediate = time.time()
    if args.print_all:
        rh_plot.plot_results([c.curve for c in inliers], data, os.path.join(args.output_path, experiment_name+'_roothairs.png'))
        rh_plot.plot_results([c.curve for c in outliers], data, os.path.join(args.output_path, experiment_name+'_outliers.png'))
        rh_density.plotDensity([rh.curve for rh in inliers], data, edge_info['closestSegments'], 
                                edge_info['edge_classes'], edge_info['edge_segments'], 
                                edge_info['edge_position'],  os.path.join(args.output_path, experiment_name+'_density.png'))

    elapsed_time = time.time() - time_intermediate
    meta_data['time_plot'] = elapsed_time
    print('Elapsed time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # *************************
    # 10. Collect Meta Data
    # *************************
    meta_data['n_roothairs'] = len(inliers)
    meta_data['n_outliers'] = len(outliers)
    
    time_total = time.time() - time_start
    meta_data['time_total'] = time_total
    print('Total time: ' + time.strftime("%H:%M:%S", time.gmtime(time_total)))

    if args.print_all:
        with open(os.path.join(args.output_path, experiment_name+'_meta.csv'), 'w') as f:
            writer = csv.writer(f)
            sortednames=sorted(meta_data.keys(), key=lambda x:x.lower())
            for name in sortednames:
                writer.writerow((name,meta_data[name]))
            sortednames=sorted(sa_parameters.keys(), key=lambda x:x.lower())
            for name in sortednames:
                writer.writerow((name,sa_parameters[name]))
            sortednames=sorted(solution_summary.keys(), key=lambda x:x.lower())
            for name in sortednames:
                writer.writerow((name,solution_summary[name]))


    print(" ")
    print("**************************************************")
    print("                    Finished!                     ")
    print("**************************************************")
    
def main():
    parser=argparse.ArgumentParser(description="Extracts and measures root hairs from classified image.")

    '''
    Input/output parameters
    '''
    parser.add_argument("-i","--in", dest="input_path", type=str, required=False,
                        default='TAKFA1-c1-1_Classes.tiff', 
                        help="tiff input file")

    parser.add_argument("-o","--out", dest="output_path", type=str, required=False,
                        default='samples/', help="csv output filename")

    parser.add_argument("-p", "--print", dest="print_all", required=False,
                        default=False, action='store_true', help="Select to output other data")

    parser.add_argument("--pixel_size", dest="pixel_size", type=float, required=False,
                        default=1, help="pixel size in microns per pixel")           # Default is 1

    '''
    Indices for root. background, root hair
    '''
    parser.add_argument("--id_root", dest="id_root", type=int, required=False,
                        default=3, help="id of root")           # Default is 1

    parser.add_argument("--id_background", dest="id_background", type=int, required=False,
                        default=2, help="id of background")     # Default is 2

    parser.add_argument("--id_roothair", dest="id_roothair", type=int, required=False,
                        default=1, help="id of root hairs")     # Default is 3

    '''
    Preprocessing parameters
    '''
    parser.add_argument("--max_dist", dest="max_dist", type=float, required=False,
                        default=10, help="Maximum distance of root hair clusters to root in pixels.")

    parser.add_argument("--thresh_d2r", dest="thresh_dist_to_root", type=int, required=False,
                        default=10, help="Minimum distance of root hair to root")

    parser.add_argument("--prune", dest="is_prune", type=bool, required=False,
                        default=True, help="Preprocessing step: Prune medial axis")

    parser.add_argument("--bin_op", dest="is_close_gaps", type=bool, required=False,
                        default=True, help="Preprocessing step: Binary opening/closing")


    '''
    Optimization parameters
    '''
    parser.add_argument("--measure", dest="measure", type=str, required=False,
                        default='strain_energy', choices=['strain_energy', 'total_curvature'], 
                        help="Type of curvature measure")
    
    parser.add_argument("--cost_type", dest="cost_type", type=str, required=False,
                        default='rms', choices=['mean', 'exp', 'rms', 'pow3', 'pow4', 'geom'], 
                        help="Way to summarize optimization objectives.")

    parser.add_argument("--n_levels", dest="n_levels", type=int, required=False,
                        default=200, help="Minimum number of iteration levels for optimization.")

    parser.add_argument('--use_ref_tips', dest='use_ref_tips', action='store_true',
                            help="Treat tips separately from non-tips for referance values.")
    parser.add_argument('--no_ref_tips', dest='use_ref_tips', action='store_false',
                            help="Do not treat tips separately from non-tips for referance values.")
    parser.set_defaults(use_ref_tips=True)

    '''
    Weights
    '''
    parser.add_argument("--w_curve", dest="w_curve", type=float, required=False,
                        default=1.0, help="Weight for curvature optimzation.")
    parser.add_argument("--w_len", dest="w_len", type=float, required=False,
                        default=1.0, help="Weight for length optimzation.")
    parser.add_argument("--w_mind", dest="w_mind", type=float, required=False,
                        default=1.0, help="Weight for minimum distance to root optimzation.")

    parser.set_defaults(func=run_pipeline)
    args=parser.parse_args()
    parser.print_help()
    parser.print_usage()
    args.func(args)



if __name__=="__main__":
    main()
