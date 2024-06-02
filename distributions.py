import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
plt.rcParams['text.usetex'] = True
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def extract_data(filename):
    file = open(filename)
    Data = []
    for line in file:
        try:
            d = np.zeros(8)
            l = line.strip().split()
            if len(l) != 0 and l[0] == 'results':
                print(l)
                d0 = l[2][2:6]
                do = d0.replace(",", "")
                d[0] = float(do)
                d[1] = float(l[3][:-1])
                d[2] = float(l[4][:-1])
                d[3] = float(l[5][:-1])
                d[4] = float(l[6][:3])
                d[5] = float(l[7])
                d[6] = float(l[8])
                d[7] = float(l[9])
                print(d)
                Data.append(d)
        except ValueError:
            continue
    return np.array(Data)

filename = "result37.out"
data = extract_data(filename)

datalog = np.log(np.log(data[:,5]))
plt.hist(data[:,6]+data[:,7], bins=int(np.sqrt(len(data[:,5])))+1, edgecolor='black')
plt.title("Amount of overlaps distribution histogram")
plt.xlabel("Amount of overlaps")
plt.ylabel('Count')
plt.grid(True)
plt.show()

"""
data[:,4] = (data[:,4]- np.mean(data[:,4]))/np.std(data[:,4])
#data[:,4] = np.log(data[:, 4])
n = len(data)

minDens = np.min(data[:, 4])
maxDens = np.max(data[:, 4])

nidx = 50
barx = np.linspace(minDens, maxDens, nidx)

def getBary(nidx, data, minDens, maxDens, n):
    idx = 0
    bary = np.zeros(nidx)
    bary[-1] += 1
    for i in range(n):
        if data[i, 4] != maxDens:
            idx = int(((data[i, 4] - minDens)/(maxDens - minDens))*nidx)
        bary[idx] += 1
    return bary

bary = getBary(nidx, data, minDens, maxDens, n)

width = np.log(30)/100
width = 30/1000
plt.figure()
plt.title("Density error distribution")
plt.bar(barx,bary, width=width)
plt.xlabel("Density error [ped/m^2]")
plt.ylabel("Count")
plt.grid(True)
plt.show()

"""