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

def getmVel(iterations):
    nPed = len(iterations[0])
    niter = len(iterations)
    dt = 1/16
    s = 0
    for i in range(nPed):
        v = 0
        k = 0
        for j in range(niter-1):
            dx = np.array([iterations[j+1,i, 0] - iterations[j,i,0], iterations[j+1,i, 1] - iterations[j,i,1]])
            vj = np.linalg.norm(dx/dt)

            if 0 < vj < 5:
                v += vj
                k += 1
                #print(vj)
        if k != 0:
            v /= k
            s += v
    return s/nPed

meanSpeed = getmVel(iterations)
print(meanSpeed)

"""
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

"""