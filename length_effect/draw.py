"""
Draw the lines of length effect
Author: Yadong Zhang
E-mail: zhangyadong@stu.xjtu.edu.cn

Demo:
$ python3 draw.py
"""
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
sys.path.append('../')
plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
from matplotlib.patches import Rectangle
from lib.visual import detectR
from lib.util import eval

# load dataset
with open('./slice/prob.npy', 'rb') as f:
    prob_res = np.load(f)
with open('./slice/time.npy', 'rb') as f:
    time_res = np.load(f)
with open('../data/ECG_X_test.bin', 'rb') as f:
    data = np.load(f)
with open('../data/ECG_info.pkl', 'rb') as f:
    label = pickle.load(f)
    label = label['Y_test']

'''Draw the length effect of an AF signal'''
idx = 18107  # an AF signal
ts = data[idx]
out = prob_res[idx]
t = time_res[idx]

fig = plt.figure(figsize=(7, 4))
gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1], hspace=0.)
# Upper panel
ax = plt.subplot(gs[0])
vertical = 1600
color = 'tab:red'
line = ax.plot(range(100, 3100, 100), 100 * np.array([p[1] for p in out]), '.r-', linewidth=1)
ax.plot([vertical, vertical], [0, 100], 'r--', linewidth=1)
ax.set_ylim([0, 100])
ax.set_ylabel('Probability of AF (\%)', color=color, fontsize=15)
ax.tick_params(axis='y', labelcolor=color)

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.spines['right'].set_color('tab:blue')
ax2.spines['left'].set_color('tab:red')
color = 'tab:blue'
# we already handled the x-label with ax1
ax2.set_ylabel('Time consumption (s)', color=color, fontsize=15)
bar = ax2.bar(x=range(100, 3100, 100), height=np.array(t), width=20, color=color, alpha=0.7)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_xlim([0, 3000])
ax2.set_xticks([])

# Lower panel
ax = plt.subplot(gs[1])
ax.plot(ts, 'k', linewidth=1)
ax.set_ylabel('ECG', fontsize=15)
ax.set_xlim([0, 3000])
ax.set_xlabel('Length of ECG signals', fontsize=15)
ax.set_ylim([-2.5, 7.5])
ax.plot([vertical, vertical], [-2.5, 7.5], 'r--', linewidth=1)
R_peak = detectR(ts)
R_peak.sort()
ax.plot(R_peak[6:8], ts[R_peak][6:8], 'r.', markersize=5)
ax.text(R_peak[6]-160, ts[R_peak][6]-2, '7th', fontsize=15)
ax.text(R_peak[7]-160, ts[R_peak][7]-2, '8th', fontsize=15)
xt = np.array(ax.get_xticks(), dtype=int)
xt = np.append(xt, vertical)
xtl = xt.tolist()
xtl.remove(1500)
xtl[-1] = str(vertical)
ax.set_xticks(np.delete(xt, 3))
ax.set_xticklabels(xtl)
ax.set_ylim([-2, 7])
boxes = [Rectangle((0, -2), 1600, 9)]
pc = PatchCollection(boxes, facecolor='red', alpha=0.2, edgecolor=None)
ax.add_collection(pc)
plt.tight_layout()
fig.savefig('flexibility.pdf', dpi=300)

'''Draw the statistical result'''
res = []
for i in range(30):
    res.append(eval(label, prob_res[:, i, 1]))
res = np.array(res)  # [30, 3]

fig = plt.figure(figsize=(7, 3))
ax = plt.subplot() 
ax.plot(np.arange(100, 3100, 100), 100 * res[:, 0], 'vy-', linewidth=1.0, markersize=6)
ax.plot(np.arange(100, 3100, 100), 100 * res[:, 1], '.r--', linewidth=1.0, markersize=4)
ax.plot(np.arange(100, 3100, 100), 100 * res[:, 2], 'og-.', linewidth=1.0, markersize=4)
ax.set_ylim([0, 100])
ax.set_xlim([0, 3000])
ax.set_xlabel('Length of ECG signals', fontsize=15)
ax.set_ylabel('Performance (\%)', fontsize=15, color='k')
ax.legend(['ROC AUC', 'PR AUC', 'F1'], fontsize=15,
          framealpha=0, loc='center')  # , ncol=3)
ax.tick_params(axis='y', labelcolor='k')
ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.spines['right'].set_color('tab:blue')
ax2.spines['left'].set_color('k')
color = 'tab:blue'
# we already handled the x-label with ax1
ax2.set_ylabel('Time consumption (s)', color=color, fontsize=15)
bar = ax2.bar(x=range(100, 3100, 100), height=np.mean(time_res, axis=0), width=20, color=color, alpha=0.7)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_xlim([0, 3000])
plt.tight_layout()
fig.savefig('length_effect.pdf', dpi=300)
