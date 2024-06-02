import numpy as np
import matplotlib.pyplot as plt

def extract_data(filename):
    file = open(filename)
    iteration = np.zeros((3005, 29, 2))
    for line in file:
        l = line.strip().split()
        Id = int(l[0])
        iter = int(l[1])

        x = -float(l[2])/100
        y = -float(l[3])/100
        position = np.array([x, y])
        iteration[iter, Id] = position
    iteration = iteration
    return iteration

filename = "eo-300-frei1/eo-300-frei1_combined_MB.txt"
iterations = extract_data(filename)

i = 0
for positions in iterations : 
    i += 1
    plt.clf()

    plt.ylim(-3.50, 3.50)
    plt.xlim(-3.50, 3.50)

    plt.scatter(positions[:, 0], positions[:, 1], color='blue')


    #plt.title(f'Iteration {t+1}')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.pause(0.005)

#plt.show()

