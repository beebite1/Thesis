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
            if euclidian_distance(densities[i], speeds[i], densities[j], speeds[j]) < epsilon:
                c+=1
        C.append(c)
    return C

def plot_fundamental_diagrams(densities, flows, speeds):
    """
    Plot fundamental diagrams: density vs flow and density vs speed.

    Args:
    - densities: Array containing density values.
    - flows: Array containing flow values.
    - speeds: Array containing speed values.
    """

    C1 = get_C(densities, flows)
    C2 = get_C(densities, speeds)

    plt.figure(figsize=(12, 5))

    n = len(densities)
    threshold = 0
    densitiesS = [densities[i] for i in range(n) if C2[i] > threshold]
    speedsS = [speeds[i] for i in range(n) if C2[i] > threshold]
    C2 = [C2[i] for i in range(n) if C2[i] > threshold]

    densitiesF = [densities[i] for i in range(n) if C1[i] > threshold]
    flowsF = [flows[i] for i in range(n) if C1[i] > threshold]
    C1 = [C1[i] for i in range(n) if C1[i] > threshold]

    plt.subplot(1,2,1)
    plt.scatter(densitiesF, flowsF, c = C1, cmap="Spectral_r")
    plt.xlabel('Density')
    plt.ylabel('Flow')
    cbar = plt.colorbar()
    cbar.set_ticks([])
    #plt.ylim(-0.1, 2.5)
    #plt.xlim(-0.1, 0.5)
    plt.title('Density vs Flow')
    
    plt.subplot(1,2,2)
    plt.scatter(densitiesS, speedsS, c = C2, cmap="Spectral_r")
    plt.xlabel('Density')
    plt.ylabel('Speed')
    cbar = plt.colorbar()
    cbar.set_ticks([])
    #plt.xlim(-0.1, 0.5)
    #plt.ylim(-0.1, 1.6)
    plt.title('Density vs Speed')

    plt.tight_layout()
    plt.show()

# Example usage
predicate = input("Paste path : ")
filename = predicate + '/fundA.txt'
pedestrian_data = read_pedestrian_data(filename)
n = len(pedestrian_data)

densities = [pedestrian_data[i]["density"] for i in range(n)]
flows = [pedestrian_data[i]["flow"] for i in range(n)]
speeds = [pedestrian_data[i]["velocity"] for i in range(n)]
plot_fundamental_diagrams(densities, flows, speeds)
