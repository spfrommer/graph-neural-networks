import os
import os.path as op
import shutil
import subprocess
import pathlib
import pickle
import json

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import pdb

sweepdir = op.join('experimentGroups', 'sweep')

def plotResults():
    with open(op.join(sweepdir, 'results.json')) as f:
        results = json.load(f)
    results = results['40']

    filter_means = [results['low']['Filter']['avg'], results['all']['Filter']['avg'], results['high']['Filter']['avg']]
    gnn_means = [results['low']['LocalGNN']['avg'], results['all']['LocalGNN']['avg'], results['high']['LocalGNN']['avg']]

    filter_std = [results['low']['Filter']['stddev'], results['all']['Filter']['stddev'], results['high']['Filter']['stddev']]
    gnn_std = [results['low']['LocalGNN']['stddev'], results['all']['LocalGNN']['stddev'], results['high']['LocalGNN']['stddev']]

    ind = np.arange(len(filter_means))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, filter_means, width, yerr=filter_std,
                    label='Filter')
    rects2 = ax.bar(ind + width/2, gnn_means, width, yerr=gnn_std,
                    label='GNN')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('MSE')
    ax.set_xticks(ind)
    ax.set_xticklabels(('Low', 'All', 'High'))
    ax.legend()

    fig.tight_layout()

    plt.show()

plot = False
if plot:
    plotResults()
    quit()

if op.isdir(sweepdir):
    shutil.rmtree(sweepdir)

nNodes = 50
#kBreak = [10, 20, 30, 40, 50]
kBreak = [40]

models = ['Filter', 'LocalGNN']

def get_results(experiment_dir):
    results = {}
    for model in models:
        results[model] = []

    evalVarsDir = op.join(experiment_dir, 'evalVars') 

    for filename in os.listdir(evalVarsDir):
        with open(op.join(evalVarsDir, filename), 'rb') as f:
            data = pickle.load(f)

        for model in models:
            if filename.startswith(model):
                results[model].append(data['costBest'])
    
    aggregate = {}

    for model in models:
        costs = results[model]
        aggregate[model] = {'avg': np.mean(costs), 'stddev': np.std(costs)}

    return aggregate

sweep_results = {}

for k in kBreak:
    k_results = {}

    lowDir = op.join(sweepdir, str(k), 'low')
    subprocess.run(['python3.7', 'subspaces.py', '--N', str(nNodes),
        '--kLow', str(0), '--kHigh', str(k), '--saveDir', lowDir])
    k_results['low'] = get_results(lowDir)

    allDir = op.join(sweepdir, str(k), 'all')
    subprocess.run(['python3.7', 'subspaces.py', '--N', str(nNodes),
        '--kLow', str(0), '--kHigh', str(nNodes), '--saveDir', allDir])
    k_results['all'] = get_results(allDir)

    highDir = op.join(sweepdir, str(k), 'high')
    subprocess.run(['python3.7', 'subspaces.py', '--N', str(nNodes),
        '--kLow', str(k), '--kHigh', str(nNodes), '--saveDir', highDir])
    k_results['high'] = get_results(highDir)

    sweep_results[k] = k_results

with open(op.join(sweepdir, 'results.json'), 'w') as f:
    json.dump(sweep_results, f, sort_keys=True, indent=4)

plotResults()

