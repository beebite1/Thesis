import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

plt.rcParams['text.usetex'] = True

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def read_pedestrian_data(filename):
    """
    Read pedestrian data from the output file.

    Args:
    - filename: Name of the output file containing pedestrian data.

    Returns:
    - List of dictionaries, each containing pedestrian information.
    """
    pedestrian_data = []
    with open(filename, 'r') as file:
        for line in file:
            try:
                data = line.strip().split()
                pedestrian_info = {
                    'timeStep': int(data[0]),
                    'measurementTime': float(data[1]),
                    'dt': float(data[2]),
                    'flow': float(data[3]),
                    'velocity': float(data[4]),
                    'density': float(data[5]),
                }
                if pedestrian_info['density'] != float('inf') and pedestrian_info['flow'] != float('inf'):
                    pedestrian_data.append(pedestrian_info)
            except:
                pass
    return pedestrian_data

def euclidian_distance(density1, speed1, density2, speed2):
    return np.sqrt((density1-density2)**2 + (speed1 - speed2)**2)

def get_C(densities, speeds, epsilon = 1e-1):
    n = len(densities)
    C = []

    for i in range(n):
        c = 0
        for j in range(n):
            if euclidian_distance(densities[i], speeds[i], densities[j], speeds[j]) < epsilon:
                c+=1
        C.append(c)
    return C

def plot_fundamental_diagrams(densities, flows, speeds, l, allData, measure):
    """
    Plot fundamental diagrams: density vs flow and density vs speed.

    Args:
    - densities: Array containing density values.
    - flows: Array containing flow values.
    - speeds: Array containing speed values.
    """

    C1 = get_C(densities, flows)
    C2 = get_C(densities, speeds)

    

    n = len(densities)
    threshold = 0
    densitiesS = [densities[i] for i in range(n) if 20 <= measure[i] <= 60]
    speedsS = [speeds[i] for i in range(n) if 20 <= measure[i] <= 60]
    #C2 = [C2[i] for i in range(n) if C2[i] > threshold]

    #densitiesF = [densities[i] for i in range(n) if C1[i] > threshold]
    #flowsF = [flows[i] for i in range(n) if C1[i] > threshold]
    #C1 = [C1[i] for i in range(n) if C1[i] > threshold]

    #plt.subplot(1,2,1)
    #plt.scatter(densitiesF, flowsF, c = C1, cmap="Spectral_r")
    #plt.xlabel('Density')
    #plt.ylabel('Flow')
    #cbar = plt.colorbar()
    #cbar.set_ticks([])
    #plt.ylim(-0.1, 2.5)
    #plt.xlim(-0.1, 0.5)
    #plt.title('Flow-Density relation')
    
    #plt.subplot(1,2,2)
    markers = ["o", "v", "^", "P", "D"]
    colors = ["blue", "red", "green", "orange"]

    labels = [1.5, 2, 2.5, 3]
    label = "exit width = " + str(labels[l-1]) + "m"
    #plt.scatter(densitiesS, speedsS, c = colors[l-1], label = label, marker=markers[l-1])
    plt.scatter(measure, densities, c = colors[l-1], label = label, marker=markers[l-1])
    allData.append([densitiesS, speedsS])
    #plt.scatter(densitiesF, flowsF, c = colors[l-1], label = l, marker=markers[l-1])


    
    

allData = []
plt.figure(figsize=(12, 5))

l = 1
 
while True:
    predicate = input("Paste path : ")
    if predicate == "End":
        break
    filename = predicate + '/fundA.txt'
    pedestrian_data = read_pedestrian_data(filename)
    n = len(pedestrian_data)

    densities = [pedestrian_data[i]["density"] for i in range(n)]
    flows = [pedestrian_data[i]["flow"] for i in range(n)]
    speeds = [pedestrian_data[i]["velocity"] for i in range(n)]
    measure = [pedestrian_data[i]["measurementTime"] for i in range(n)]
    
    plot_fundamental_diagrams(densities, flows, speeds, l, allData, measure)
    l += 1

finalData = [[], []]

for i in range(len(allData)):
    j = len(allData[i][0])
    for k in range(j):
        finalData[0].append(allData[i][0][k])
        finalData[1].append(allData[i][1][k])
finalData = np.array(finalData)

final0 = finalData[0][finalData[0] != 0]
final1 = finalData[1][finalData[1] != 0]

final = np.array([final0, final1])

Xf = final[0]
yf = final[1]
slope, intercept, r_value, p_value, std_err = linregress(Xf, yf)


X = np.linspace(0, 6, 100)
Y = -0.204*X + 1.48
Yf = slope*X + intercept

#plt.plot(X, Y, label = "Speed-Density relation according to (Mōri and Tsukaguchi, 1987)", color = "black")
#plt.plot(X, Yf, label = "Linear regression of the data")
plt.ylabel('Density $[m^{-2}]$')
plt.xlabel('Time [s]')
#plt.ylabel('Flow [1/(m s)]')
plt.axvline(x = 25, color = 'black', label = 'Stationary state interval')
plt.axvline(x = 65, color = 'black')
plt.title('Density over time')
#plt.title('Flow-Density relation')

#plt.xlim(right = 6)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
