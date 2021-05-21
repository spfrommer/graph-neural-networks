import matplotlib.pyplot as plt
import matplotlib
import scipy.stats
import numpy as np
import ast
import pdb

lines = False

matplotlib.rcParams.update({'font.size': 22})

x = np.array([0,1,2,3,4,5,6])

my_xticks = ['Input','Filter','ReLU','Filter','ReLU','Filter','ReLU']
plt.xticks(x, my_xticks)
plt.ylabel('Separability (0 = perfectly separable, 50 = inseparable)')

if lines:
    with open('results.txt') as f:
        for line in f:
            y = ast.literal_eval(line)
            plt.plot(x, y, color='cornflowerblue', linewidth=3)
else:
    def confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        return se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    with open('results_standard.txt') as f:
        ys = [ast.literal_eval(line) for line in f]
        means = np.mean(np.array(ys), axis=0)
        confidence = [confidence_interval(vals) for vals in np.array(ys).T.tolist()]
        plt.errorbar(x, means, confidence, linewidth=3, label='standard basis')
        #plt.plot(x, np.mean(np.array(ys), axis=0), linewidth=3, label='standard basis')

    with open('results_eigenbasis.txt') as f:
        ys = [ast.literal_eval(line) for line in f]
        means = np.mean(np.array(ys), axis=0)
        confidence = [confidence_interval(vals) for vals in np.array(ys).T.tolist()]
        plt.errorbar(x, means, confidence, linewidth=3, label='eigen basis')
        #plt.plot(x, np.mean(np.array(ys), axis=0), linewidth=3, label='eigenbasis')

plt.legend()
plt.show()
