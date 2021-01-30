import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import gridspec 
plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
import matplotlib.patches as patches
from tqdm import tqdm

from scipy.signal import butter, lfilter
ECG_param = {'PQ_max': 0.2, 'QRS_max': 0.150, 
            'P_min': 0.08, 'RR_min': 0.200,
            'sample_freq': 300
            } # unit: s, Hz

def detectR(data, threshold=0.96): 
    '''
    detect R peaks from ECG signals
    :param data: 1-d array, ECG signals
    :param threshold: float, threshold for the R peaks
    return: R peak indices
    '''
    RR_interval = int(ECG_param['RR_min']*ECG_param['sample_freq'])
    # Filter the QRS regions for R detection
    nyquist_freq = 0.5 * ECG_param['sample_freq']
    bandpass = [10, 50]  # R band
    low = bandpass[0] / nyquist_freq
    high = bandpass[1] / nyquist_freq
    filter_order = 1
    b, a = butter(filter_order, [low, high], btype="band")
    ts = lfilter(b, a, data)
    
    # Second order difference of the filtered time series
    diff2ts = np.power(np.diff(np.diff(ts)), 2)
    # Find the top 96% amplitude of the ECG as the threshold of R peaks
    thres_amp= np.percentile(diff2ts, int(threshold*100))
    
    idxsort = np.argsort(diff2ts)  # obtain the sorted idx of time series
    idx = np.arange(len(idxsort))
    filtered = []
    for i in idxsort[::-1]:
        if diff2ts[i] > thres_amp:
            if i in idx:
                filtered.append(i)  # if i is in the idx, it means that the peak is not removed previously
            idx = np.setdiff1d(idx, range(i-RR_interval//2, i+RR_interval//2))  # remove the idx
    R_peak = []
    for i in filtered:
        try:
            p = i-RR_interval//2 + np.argmax(data[i-RR_interval//2: i+RR_interval//2])
            if np.max(data[i-RR_interval//2: i+RR_interval//2]) > np.mean(data):
                R_peak.append(p)
        except:
            pass

    return R_peak

def flip_ECG(data):
    '''
    Some ECG is reversed which needed to be inversed
    :param data: 1-d array, ECG signal
    return: ECG signal
    '''
    less, more = np.percentile(data, (2, 98))
    if np.abs(less) > np.abs(more):
        return -data
    else:
        return data

def get_beats(data, R_peak:list):
    '''
    Get beats
    :param ts: 1-d array, ECG signals
    :param R_peak: list, R_peak indices
    return: list of beats
    '''
    R_peak = np.sort(R_peak)
    RR_interval = int(np.mean(np.diff(R_peak)))
    beats = []
    for i in R_peak:
        beats.append(data[i-RR_interval//2: i+RR_interval//2])
    return beats
    
def filt_ECG(data, freq='mid', sample_freq=300):
    '''
    filter the ECG signal
    :param freq: str, 'mid', 'high', 'low', 'no'
    :param sample_freq: int
    return: filtered ECG signal
    '''
    if freq == 'no': return data
    nyquist_freq = 0.5 * sample_freq
    select = {'low': (0.001 / nyquist_freq, 0.5 / nyquist_freq), 'mid': (0.5 / nyquist_freq, 50 / nyquist_freq), 'high': 50 / nyquist_freq}
    filter_order = 1
    if freq == 'high':
        b, a = butter(filter_order, select[freq], btype="high")
    else:
        b, a = butter(filter_order, select[freq], btype="band")
    out = lfilter(b, a, data)
    return out

def argmax2D(matrix):
    """
    find the index of maximum value in 2d matrix
    :param matrix: 2-d array
    :return: x, y
    """
    cols = matrix.shape[1]
    loc = np.argmax(matrix)
    y, x = loc //cols,loc %cols
    return y, x

def plot_cam(ts, img, path=None, point=None, peak_idx=None):
    '''
    plot time series and SG-CAM images
    :param ts: 1-d array, ECG signal
    :param img: 2-d array, image of SG-CAM
    :param path: str, path to save the figure
    :param point: plot which point in the SG-CAM
    :param peak_idx: plot which peak in ECG
    '''
    if point:
        win = 100
        centerx, centery = argmax2D(img[point[1]-win:point[1]+win, point[0]-win: point[0]+win])
        centerx += point[1]-win
        centery += point[0]-win
        
    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1,4], hspace=0.) 
    ax = plt.subplot(gs[0]) 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([0, 3000-3])
    ax.plot(ts, 'k', linewidth=1)
    if point:
        motif = [
             [centery, centerx+1, 'ro--', 4]
        ]
        for start, gap, ty, s in motif:
            ix = np.arange(start, start + gap*3, gap)
            ax.plot(ix, ts[ix], ty, markersize=s)

    ax.set_ylim([-4, 8.5])
    ts_flip = flip_ECG(ts)
    R_peak = detectR(ts_flip)
    R_peak.sort()
    for i in R_peak:
        ax.plot([i, i], [-4, 8.5], 'r--', linewidth=0.5)

    ax = plt.subplot(gs[1])
    ax.imshow(img, cmap=plt.cm.jet, vmax=1, vmin=0)
    if point:
        ax.plot([centery], [centerx], 'r+', markersize=10, alpha=0.5)
        
    horizon_move = 100
    if peak_idx:
        rect = patches.Rectangle([R_peak[peak_idx]-50, 0],50*2,30*2,linestyle='--',linewidth=1,edgecolor='w',facecolor='none')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.1)
        ax.text(R_peak[peak_idx]+horizon_move, 160, "QRS", color='w', ha="center", va="center", size=14,
                bbox=bbox_props)
        ax.add_patch(rect)

        rect = patches.Rectangle([R_peak[peak_idx]-170, 0],50*2,30*2,linestyle='--',linewidth=1,edgecolor='w',facecolor='none')
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.1)
        ax.text(R_peak[peak_idx]-250+horizon_move, 160, "P", color='w', ha="center", va="center", size=14,
                bbox=bbox_props)
        ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=300)

def plot_basic_tsne(X_embedded, val_y, alpha=1, s=1, path=None):
    '''
    draw tsne figure of AF and non-AF
    :param X_embedded: 2-d array, shape is [N, 2], the low-dimensional points
    :param val_y: 1-d array, shape is [N], the label of the data, 1 and 0 indicated the AF and non-AF
    :param alpha: float, alpha of the scatters
    :param s: float, size of the scatters
    :param path: str, path of saving the figure
    '''
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x=X_embedded[val_y==1][:,0], y=X_embedded[val_y==1][:,1],color='r',s=s,alpha=alpha,edgecolors="none")
    plt.scatter(x=X_embedded[val_y==0][:,0], y=X_embedded[val_y==0][:,1],color='b',s=s,alpha=alpha,edgecolors="none")
    lgnd = plt.legend(['AF', 'non-AF'],fancybox=False, framealpha=0,fontsize=15, loc='lower left')

    for handle in lgnd.legendHandles:
        handle.set_sizes([12.0])
        handle.set_alpha(1.0)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300)

def plot_AF_pid(X_embedded, val_y, pid_test, path=None):
    '''
    draw tsne figure of AF according to the id of patients (pid)
    :param X_embedded: 2-d array, shape is [N, 2], the low-dimensional points
    :param val_y: 1-d array, shape is [N], the label of the data, 1 and 0 indicated the AF and non-AF
    :param pid_test: 1-d array, shape is [N], the id of the patients
    :param path: str, path of saving the figure
    '''
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x=X_embedded[val_y==1][:,0], y=X_embedded[val_y==1][:,1],
                    s=0.5,alpha=1,c=pid_test[val_y==1], cmap='jet')

    x, y, s = X_embedded[val_y==1][:,0], X_embedded[val_y==1][:,1], pid_test[val_y==1]
    for i in tqdm(np.unique(s)):
        for xi, yi in zip(x[s==i], y[s==i]):
            plt.plot([np.mean(x[s==i]), xi], [np.mean(y[s==i]), yi], 'k', linewidth=0.1, alpha=0.2)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300)
