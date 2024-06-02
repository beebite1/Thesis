import numpy as np
import matplotlib.pyplot as plt

def extract_data(filename):
    file = open(filename)
    iteration = np.zeros((1150, 353, 2))
    for line in file:
        l = line.strip().split()
        Id = int(l[0])
        iter = int(l[1])

        x = -float(l[3])/100
        y = float(l[2])/100
        position = np.array([x, y])
        iteration[iter, Id] = position
    return iteration

filename = "AO/ao-240-400_combine.txt"
iterations = extract_data(filename)
print(np.min(iterations[:,:, 1]))

"""
i = 0
for positions in iterations : 
    i += 1
    plt.clf()

    plt.ylim(-2.50, 4.5)
    plt.xlim(-8.5, 7)

    plt.scatter(positions[:, 0], positions[:, 1], color='blue')


    #plt.title(f'Iteration {t+1}')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid()
    plt.pause(0.005)

#plt.show()
"""

def getDensity(iterations):
    x0 = -2.4
    x1 = 2.4
    y0 = -0.2
    y1 = 2.2
    Area = 2.4*4.8
    densities = []
    for positions in iterations:
        c = 0
        for i in range(len(positions)):
            if x0 < positions[i, 0] < x1 and y0 < positions[i, 1] < y1 and (positions[i, 0] != 0 or positions[i, 1] != 0):
                if positions[i,0] == 0 :
                    print(positions[i])
                c += 1
        densities.append(c/(Area))
    return densities
    
density = getDensity(iterations)

meanD = np.mean(np.array(density[200:800]))
print(meanD)
plt.figure()
plt.plot(density)
plt.savefig("density.png")
plt.show()
