import numpy as np
import matplotlib.pyplot as plt

def extract_data(filename):
    file = open(filename)
    iteration = np.zeros((1200, 353, 2))
    lastId = 0
    copyD = False
    for line in file:
        l = line.strip().split()
        Id = int(l[0])
        iter = int(l[1])

        if Id > lastId and iter == 30:
            lastId = Id
            copyD = True

        if Id > lastId and iter != 30:
            copyD = False

        if copyD:
            x = -float(l[3])/100
            y = float(l[2])/100
            position = np.array([x, y])
            iteration[iter, Id] = position
    iteration = iteration[30:100]
    return iteration

filename = "eo-300-frei1/eo-300-frei1_cam1_MB.txt"
iterations = extract_data(filename)

print(iterations[0])
i = 0
for positions in iterations : 
    if i > 100 : break
    i += 1
    plt.clf()

    plt.ylim(-2.50, 4.50)
    plt.xlim(-8.00, 7.00)

    plt.scatter(positions[:, 0], positions[:, 1], color='blue')


    #plt.title(f'Iteration {t+1}')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.pause(0.005)

#plt.show()

