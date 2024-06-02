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
                    'velocity': float(data[1]),
                    'density': float(data[2]),
                }
                if pedestrian_info['density'] != float('inf') :
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

def plot_fundamental_diagrams(densities, speeds):
    """
    Plot fundamental diagrams: density vs flow and density vs speed.

    Args:
    - densities: Array containing density values.
    - flows: Array containing flow values.
    - speeds: Array containing speed values.
    """

    C = get_C(densities, speeds)

    plt.figure()

    plt.scatter(densities, speeds, c = C, cmap="Spectral_r")
    plt.xlabel('Density')
    plt.ylabel('Speed')
    #plt.xlim(-0.1, 1.2)
    #plt.ylim(-0.1, 1.6)
    plt.title('Density vs Speed')
    cbar = plt.colorbar()
    cbar.set_ticks([])

    plt.tight_layout()
    plt.show()

# Example usage
predicate = input("Paste path : ")
filename = predicate + '/fundC.txt'
pedestrian_data = read_pedestrian_data(filename)
n = len(pedestrian_data)

densities = [pedestrian_data[i]["density"] for i in range(n)]
speeds = [pedestrian_data[i]["velocity"] for i in range(n)]
plot_fundamental_diagrams(densities, speeds)
