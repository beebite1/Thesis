import numpy as np
import matplotlib.pyplot as plt
import copy
from fteikpy import Eikonal2D

def extract_data(filename):
    file = open(filename)
    iteration = np.zeros((1150, 353, 2))
    idx = []
    for line in file:
        l = line.strip().split()
        Id = int(l[0])
        iter = int(l[1])

        x = -float(l[3])/100
        y = float(l[2])/100
        position = np.array([x, y])
        iteration[iter, Id] = position
    for i in range(len(iteration[0])):
        if -2.4 < iteration[30, i, 0] < 0 and 0 < iteration[30, i, 1] < 2.4:
            idx.append(i)
    return iteration[:,idx]

filename = "AO/ao-240-400_combine.txt"
iterations = extract_data(filename)

def floorfield(positions, room_width, room_height, deleted):
    numPed = len(positions)
    allGrads = np.zeros((numPed,2))
    i = 0
    for position in positions:
        if deleted[i] == -1: 
            resol = 1e-2
            x = position[0]
            y = position[1]

            x = int((position[0]+8)/resol)
            y = int((position[1]+2.2)/resol)

            # Velocity model


            n = int(room_height/resol)
            m = int(room_width/resol)

            wallval = 1e-2
            velocity_model = np.ones((m, n))
            #velocity_model[0,:] = wallval
            #velocity_model[:,n-1] = wallval
            velocity_model[int(8/resol):int(9/resol),:int(1.7/resol)] = wallval
            velocity_model[int(8/resol):int(9/resol),int(4.1/resol):] = wallval
            #velocity_model[3:,3] = wallval

            dx, dy = resol, resol

            # Solve Eikonal at source
            eik = Eikonal2D(velocity_model, gridsize=(dx, dy))
            tt = eik.solve((14.3, 2), return_gradient=True)

            tgradx = tt.gradient[0][x][y]
            tgrady = tt.gradient[1][x][y]

            grad = [-tgradx, -tgrady]
            allGrads[i] = grad
        i+=1
    return allGrads

def get_vect(position, exit_doorx, exit_doory):
    distances = np.sqrt((exit_doorx - position[0])**2 + (exit_doory - position[1])**2)
    minDidx = np.argmin(distances)
    minD = distances[minDidx]

    dx = exit_doorx[minDidx] - position[0]
    dy = exit_doory[minDidx] - position[1]

    res = np.array([dx, dy])
    vect = res/(np.linalg.norm(res)+1e-5)
    return vect, minD

def is_in_array(position, deleted):
    for elem in deleted:
        compx = position[0] == elem[0]
        compy = position[1] == elem[1]
        if compx and compy:
            return True
        if np.equal(position, elem).all():
            print(np.equal(position, elem).all())
            return True
    return False

def count_overlaps(positions, xstart, xend, r):
    count = 0
    n = len(positions)
    for i in range(n):
        x = positions[i]
        for j in range(n):
            if i != j:
                y = positions[j]
                d = np.linalg.norm(x - y)
                if d < r[i]+r[j] and xstart < x[0] < xend and xstart < y[0] < xend:
                    count += 1
    return count

def wall_overlaps(positions, wallsx, wallsy, r):
    count = 0
    n = len(positions)
    for i in range(n):
        __, minD = get_vect(positions[i], wallsx, wallsy)
        if minD < r[i]:
            count += 1
    return count 

def still_in(positions, xstart, xend):
    count = 0
    for pos in positions:
        if xstart < pos[0] < xend:
            count += 1
    return count

def social_force_vadere(A, B, C, D,  kn, kt, room_width, room_height, door_width, exit_doorx,
              exit_doory, 
              num_pedestrians, num_iterations, dt, deleted,
              tau, m, v0, l, positions, wallsy, wallsx,
              velocities, r):
    forces = np.zeros_like(velocities)
    eikDir = floorfield(positions, room_width, room_height, deleted)
    
    for i in range(num_pedestrians):
        if deleted[i] != -1 :
                continue

        pos = positions[i]
        
        __, minD = get_vect(pos, exit_doorx, exit_doory)
        ed = eikDir[i]
        Fd = (v0*ed - velocities[i])/tau
        forces[i] += Fd

        ew, minW = get_vect(pos, wallsx, wallsy)
        ew = -ew
        Fw =  C* np.exp((r[i]*0 - minW) / D) * ew
        forces[i] += Fw

        for j in range(num_pedestrians):

            if deleted[j] != -1 :
                continue

            if i != j:
                rij = r[i] + r[j]
                xdij = positions[i] - positions[j]
                dij = np.linalg.norm(xdij)
                nij = xdij / dij

                ei = velocities[i]/np.linalg.norm(velocities[i])
                #ej = velocities[j]/np.linalg.norm(velocities[j])

                cosphi = -np.dot(nij, ei)
                b = 0.5*np.sqrt((dij + np.linalg.norm(xdij - velocities[j]*dt))**2 - (np.linalg.norm(velocities[j])*dt)**2)
                Fi =  A* np.exp((rij*0 -b) / B) * nij * (l + (1-l)*((1+cosphi)/2))
                forces[i] += Fi

        if minD <= 0.1 :
            deleted[i] = i
    return forces

def realDist(positions, iterations):
    res = np.sqrt((positions[:,0]-iterations[:,0])**2 + (positions[:,1]-iterations[:,1])**2)
    return res

def main_loop(A, B, C, D, kn, kt, room_width, room_height, door_width, exit_doorx,
              exit_doory, 
              num_pedestrians, num_iterations, dt, deleted,
              tau, m, v0, l, positions, wallsy, wallsx,
              velocities, r, iterations, model = social_force_vadere):
    
    # Metrics initialization
    total_overlaps = 0
    walls_overlaps = 0
    distanceNorm = 0

    deleted = np.arange(num_pedestrians)

    # Simulation loop
    for t in range(num_iterations):
        if t < 30:
            continue

        if (t%100 == 0):
            print(deleted)

        

        for pedId in range(num_pedestrians):
            if deleted[pedId] == pedId and iterations[t, pedId, 0] != 0 and iterations[t, pedId, 1] != 0:
                deleted[pedId] = -1
                positions[pedId] = iterations[t, pedId]
        
        # If all arrive, finish simulation
        #if (deleted == np.arange(num_pedestrians)).all():
        #    break
        
        # Get all forces exerted on pedestrians
        forces = model(A, B, C, D, kn, kt, room_width, room_height, door_width, exit_doorx,
              exit_doory, 
              num_pedestrians, num_iterations, dt, deleted,
              tau, m, v0, l, positions, wallsy, wallsx,
              velocities, r)

        # F = ma (and motion equations)
        accelerations = forces / m
        velocities += accelerations * dt
        
        # If pedestrian arrived, he doesn't move anymore
        for i in range(len(velocities)):
            if deleted[i] != -1:
                velocities[i] = np.array([1e-8,1e-8])
        positions += velocities * dt

        # Count metrics
        rDist = realDist(positions, iterations[t])
        distanceNorm += np.linalg.norm(rDist)

        total_overlaps += count_overlaps(positions, room_width/2, room_width - 0.1, r)
        if (deleted != np.arange(num_pedestrians)).any():
            walls_overlaps += wall_overlaps(positions, wallsx, wallsy, r)

        
        plt.clf()
        plt.ylim(-2.2, 4.1)
        plt.xlim(-8, 6.3)

        #for i in range(len(deleted)):
        #    if deleted[i] == -1:
        plt.scatter(positions[:, 0], positions[:, 1], color='blue', label = "SFM pedestrian position")
        plt.scatter(iterations[t, :, 0], iterations[t, :, 1], color='red', label = "Measured pedestrian position")
        
        plt.scatter(wallsx, wallsy, color='black', s = 5)
        plt.scatter(exit_doorx, exit_doory, color='green', s = 20)
        plt.title(f'Iteration {t+1}')
        plt.legend()
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.pause(0.01)
        
    
    plt.show()
    

    ped_left = still_in(positions, xstart=room_width/2, xend=room_width - 0.2)

    return distanceNorm



def grid_search(iterations):
    iter0 = copy.deepcopy(iterations[0])
    # Constants
    room_width = 15.5  # meters
    room_height = 7  # meters
    door_width = 2.4
    exit_doorx = np.linspace(0 , door_width, 100)
    exit_doory = -4*np.ones(100)
    num_pedestrians = len(iterations[0])
    num_iterations = 3005
    dt = 0.2  # Time step
    dt = 1/16
    deleted = -1*np.ones(num_pedestrians)

    # Parameters
    tau = 0.5  # Relaxation time

    gridResol = 3
    TAU = np.linspace(0.25, 0.75, gridResol)
    V0 = np.linspace(1.2, 1.6, gridResol)
    m = 1  # Pedestrian mass
    v0 = 1.34  # Desired velocity 

    l = 0.61

    # Initial positions and velocities of pedestrians
    #positions = np.random.rand(num_pedestrians, 2) * np.array([room_width/2, room_height])

    positions = iter0

    nw = 1000

    wallsy1 = 3*np.ones(nw)
    wallsx1 = np.linspace(-4, 3, nw)

    wallsy2 = np.linspace(3, -4, nw)
    wallsx2 = 3*np.ones(nw)

    wallsy3 = np.zeros(nw)
    wallsx3 = np.linspace(-4, 0, nw)

    wallsy4 = np.linspace(0, -4, nw)
    wallsx4 = np.zeros(nw)

    wallsyg1 = np.append(wallsy1, wallsy2)
    wallsyg2 = np.append(wallsy3, wallsy4)
    wallsy = np.append(wallsyg1, wallsyg2)

    wallsxg1 = np.append(wallsx1, wallsx2)
    wallsxg2 = np.append(wallsx3, wallsx4)
    wallsx = np.append(wallsxg1, wallsxg2)

    velocities = np.ones((num_pedestrians, 2))*v0

    r = 0.2*np.ones((num_pedestrians)) # Pedestrian radius

    A = 4.85  # Interaction strength
    B = 0.68 # Interaction range
    C = 17.69
    D = 0.21

    A = 0.1  # Interaction strength
    B = 0.1 # Interaction range
    C = 0.1
    D = 0.1

    kn = 2
    kt = 2

    minRealDist = float('inf')
    minTau = float('inf')
    minv0 = float('inf')

    top5 = []

    i = 0
    j = 0
    for tau in TAU:
        for v0 in V0:
            j+=1
            if j%4 == 1:
                suffix = "/"
            elif j%4 == 2:
                suffix = "-"
            elif j%4 == 3:
                suffix = "\\"
            else:
                    suffix = "|"
            print("", end=f"\r{suffix} Best parameters found yet : {minTau, minv0} | Result : {round(minRealDist,2)} | Top5 : {top5} | PercentComplete: {round(i,2)} %")
            i += 1/gridResol**2*100


            deleted = -1*np.ones(num_pedestrians)
            positions = iter0
            velocities = np.ones((num_pedestrians, 2))*v0

            realDistNorm = main_loop(A, B, C, D, kn, kt, room_width, room_height, door_width, exit_doorx,exit_doory, num_pedestrians, num_iterations, dt, deleted, tau, m, v0, l, positions, wallsy, wallsx, velocities, r, iterations)
            if realDistNorm < minRealDist:
                minRealDist = realDistNorm
                minTau = tau
                minv0 = v0
                top5.append(minTau)
                if len(top5) > 5:
                    top5.pop(0)
                    

    return minTau

#minTau = grid_search(iterations)
#print("\ntau = ", minTau)

A = 0.0  # Interaction strength
B = 0.1 # Interaction range
C = 0.0
D = 0.1

A = 4.85  # Interaction strength
B = 0.68 # Interaction range
C = 17.69
D = 0.21

A = 2.1  # Interaction strength
B = 0.3 # Interaction range
C = 10
D = 0.2




iter0 = copy.deepcopy(iterations[0])

# Constants
room_width = 14.3  # meters
room_height = 6.3  # meters
door_width = 3
exit_doorx = 6.3*np.ones(100)
exit_doory = np.linspace(-2.2, 4.1, 100)
num_pedestrians = len(iterations[0])
num_iterations = 1150
dt = 1/16 # Time step
deleted = -1*np.ones(num_pedestrians)

# Parameters
tau = round(10 * (5/16),3)  # Relaxation time

m = 1  # Pedestrian mass
v0 = 1.34  # Desired velocity 
#v0 = 1.5551577064433808

l = 0.61


# Initial positions and velocities of pedestrians
positions = iter0

nw = 1000

wallsy1 = np.linspace(-2.2, 0, nw)
wallsx1 = np.zeros(nw)

wallsy2 = np.linspace(2.4, 4.1, nw)
wallsx2 = np.zeros(nw)


wallsy = np.append(wallsy1, wallsy2)

wallsx = np.append(wallsx1, wallsx2)

velocities = np.ones((num_pedestrians, 2))*v0 * [1, 0]


r = 0.2*np.ones((num_pedestrians)) # Pedestrian radius
main_loop(A, B, C, D, 2, 2, room_width, room_height, door_width, exit_doorx, exit_doory, num_pedestrians, num_iterations, dt, deleted, tau, m , v0, l, positions, wallsy, wallsx, velocities, r, iterations)



