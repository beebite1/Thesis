import numpy as np
import matplotlib.pyplot as plt

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
            if np.abs(speeds[i] - speeds[j]) < epsilon:
                c+=1
        C.append(c)
    return C

def plot_fundamental_diagrams(flows, timeStep, flows2, timeStep2):
    """
    Plot fundamental diagrams: density vs flow and density vs speed.

    Args:
    - densities: Array containing density values.
    - flows: Array containing flow values.
    - speeds: Array containing speed values.
    """

    C1 = get_C(timeStep, flows)
    C2 = get_C(timeStep2, flows2)

    plt.figure(figsize=(12, 5))

    n = len(timeStep2)
    threshold = 3
    flowsS = [flows2[i]*22 for i in range(n) if C2[i] > threshold]
    timeStepS = [timeStep2[i] for i in range(n) if C2[i] > threshold]
    C2 = [C2[i] for i in range(n) if C2[i] > threshold]

    n = len(timeStep)
    timeStepF = [timeStep[i] for i in range(n) if C1[i] > threshold]
    flowsF = [flows[i]*14 for i in range(n) if C1[i] > threshold]
    C1 = [C1[i] for i in range(n) if C1[i] > threshold]

    plt.subplot(1,2,1)
    plt.scatter(timeStepF, flowsF, c = C1, cmap="Spectral_r")
    plt.xlabel('Time Step')
    plt.ylabel('Flow [1/s]')
    cbar = plt.colorbar()
    cbar.set_ticks([])
    #plt.ylim(-0.1, 2.5)
    #plt.xlim(-0.1, 0.5)
    plt.title('Flow over time Shengen')
    
    plt.subplot(1,2,2)
    plt.scatter(timeStepS, flowsS, c = C2, cmap="Spectral_r")
    plt.xlabel('Time Step')
    plt.ylabel('Flow [1/s]')
    cbar = plt.colorbar()
    cbar.set_ticks([])
    #plt.xlim(-0.1, 0.5)
    #plt.ylim(-0.1, 1.6)
    plt.title('Flow over time Non-Shengen')

    plt.tight_layout()
    plt.show()

# Example usage
predicate = input("Paste path : ")
filename = predicate + '/fundA1.txt'
pedestrian_data = read_pedestrian_data(filename)
n = len(pedestrian_data)

densities = [pedestrian_data[i]["density"] for i in range(n)]
flows = [pedestrian_data[i]["flow"] for i in range(n)]
timeStep = [pedestrian_data[i]["timeStep"] for i in range(n)]

filename2 = predicate + '/fundA2.txt'
pedestrian_data2 = read_pedestrian_data(filename2)
n = len(pedestrian_data2)

densities2 = [pedestrian_data2[i]["density"] for i in range(n)]
flows2 = [pedestrian_data2[i]["flow"] for i in range(n)]
timeStep2 = [pedestrian_data2[i]["timeStep"] for i in range(n)]


plot_fundamental_diagrams(flows, timeStep, flows2, timeStep2)
