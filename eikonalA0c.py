import numpy as np
import matplotlib.pyplot as plt
import copy
from fteikpy import Eikonal2D

#plt.rcParams['text.usetex'] = True

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
    return iteration

filename = "AO/ao-240-400_combine.txt"
iterations = extract_data(filename)

def wold(e, f):
    if np.dot(e, f) >= np.linalg.norm(f)*np.cos(np.pi*100/180):
        return 1
    return 0.5
def getDensity(iterations):
    x0 = -2.4
    x1 = -1e-3
    x2 = 1e-3
    x3 = 2.4
    y0 = -0.2
    y1 = 2.2
    Area = 2.4*4.8
    densities = []
    for positions in iterations:
        c = 0
        for i in range(len(positions)):
            if (x0 < positions[i, 0] < x1 and y0 < positions[i, 1] < y1) or (x2 < positions[i, 0] < x3 and y0 < positions[i, 1] < y1):
                c += 1
        densities.append(c/(Area))
    return np.array(densities)

def getmyDensity(positions, myDens):
    x0 = -2.4
    x1 = -1e-3
    x2 = 1e-3
    x3 = 2.4
    y0 = -0.2
    y1 = 2.2
    Area = 2.4*4.8
    c = 0
    for i in range(len(positions)):
        if (x0 < positions[i, 0] < x1 and y0 < positions[i, 1] < y1) or (x2 < positions[i, 0] < x3 and y0 < positions[i, 1] < y1):
            c += 1
    myDens.append(c/(Area))
    return

def getDensityError(densities, myDens):
    error = 0
    for i in range(50, len(myDens)-400):
        error += np.abs(densities[i] - myDens[i])
    return error

def getGrad(positions, deleted, tt):
    numPed = len(positions)
    allGrads = np.zeros((numPed,2))
    ttx = int(np.shape(tt)[0])
    tty = int(np.shape(tt)[1])
    i = 0
    resol = 1e-2
    for position in positions:
        if deleted[i] == -1: 
            x = int((position[0]+8)/resol)
            y = int((position[1]+2.2)/resol)

            if x >= ttx:
                x = ttx - 1
            if x < 0:
                x = 0
            if y >= tty:
                y = tty - 1
            if y < 0:
                y = 0

            tgradx = tt.gradient[0][x][y]
            tgrady = tt.gradient[1][x][y]

            grad = [-tgradx, -tgrady]
            allGrads[i] = grad
        i+=1
    return allGrads

def floorfield(room_width, room_height):
    resol = 1e-2

    # Velocity model
    n = int(room_height/resol)
    m = int(room_width/resol)

    wallval = 1e-2
    velocity_model = np.ones((m, n))
    velocity_model[0,:] = wallval
    velocity_model[m-1,:] = wallval
    velocity_model[:,n-1] = wallval
    velocity_model[:,0] = wallval

    velocity_model[int(8/resol):int(9/resol),:int(1.9/resol)] = wallval
    velocity_model[int(8/resol):int(9/resol),int(4.3/resol):] = wallval
    #velocity_model[3:,3] = wallval

    dx, dy = resol, resol

    # Solve Eikonal at source
    eik = Eikonal2D(velocity_model, gridsize=(dx, dy))
    tt = eik.solve((14.3, 2), return_gradient=True)

    return tt

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

def count_overlaps(positions, r, deleted):
    count = 0
    n = len(positions)
    for i in range(n):
        if deleted[i] == -1:
            x = positions[i]
            for j in range(n):
                if i != j and deleted[j] == -1:
                    y = positions[j]
                    d = np.linalg.norm(x - y)
                    if d < r[i]+r[j]:
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

def social_force_vadere(A, B, C, D, exit_doorx,
              exit_doory, 
              num_pedestrians, dt, deleted,
              tau, v0, l, positions, wallsy, wallsx,
              velocities, r, tt, Helbing = False):
    forces = np.zeros_like(velocities)
    
    eikDir = getGrad(positions, deleted, tt)
    Dt = 2
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
                ej = velocities[j]/np.linalg.norm(velocities[j])
                d2 = np.linalg.norm(xdij - velocities[j]*Dt)
                cosphi = -np.dot(nij, ei)
                b = 0.5*np.sqrt((dij + np.linalg.norm(xdij - velocities[j]*Dt))**2 - (np.linalg.norm(velocities[j])*Dt)**2)
                #Fi =  A* np.exp((rij*0 -b) / B) * nij * (l + (1-l)*((1+cosphi)/2))
                if Helbing == False:
                    Fi =  A* np.exp((-b) / B) * (l + (1-l)*((1+cosphi)/2)) * (dij + d2)/(2*b) * (xdij/dij + (xdij - velocities[j]*Dt)/d2)/2
                    forces[i] += Fi
                else : 
                    Fi =  A* np.exp((-b) / B) * (dij + d2)/(2*b) * (xdij/dij + (xdij - velocities[j]*Dt)/d2)/2
                    w = wold(ei, -Fi)
                    forces[i] += w*Fi

        if minD <= 0.1 :
            deleted[i] = -2
    return forces

def realDist(positions, iterations):
    res = np.sqrt((positions[:,0]-iterations[:,0])**2 + (positions[:,1]-iterations[:,1])**2)
    return res

def main_loop(A, B, C, D, kn, kt, room_width, room_height, door_width, exit_doorx,
              exit_doory, 
              num_pedestrians, num_iterations, dt, deleted,
              tau, m, v0, l, positions, wallsy, wallsx,
              velocities, r, iterations, model = social_force_vadere, Helbing = False):
    
    # Metrics initialization
    total_overlaps = 0
    walls_overlaps = 0
    distanceNorm = 0

    
    deleted = np.arange(num_pedestrians)
    tt = floorfield(room_width, room_height)

    myDens = []
    # Simulation loop
    for t in range(num_iterations):
        if t < 43:
            continue

        if (t%10 == 0):
            print(t)
            #print(deleted)

        

        for pedId in range(num_pedestrians):
            if deleted[pedId] == pedId and iterations[t, pedId, 0] != 0 and iterations[t, pedId, 1] != 0:
                deleted[pedId] = -1
                positions[pedId] = iterations[t, pedId]
        
        # If all arrive, finish simulation
        #if (deleted == np.arange(num_pedestrians)).all():
        #    break
        
        # Get all forces exerted on pedestrians
        forces = model(A, B, C, D, exit_doorx,
              exit_doory, 
              num_pedestrians, dt, deleted,
              tau, v0, l, positions, wallsy, wallsx,
              velocities, r, tt, Helbing = Helbing)

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

        total_overlaps += count_overlaps(positions, r, deleted)
        if (deleted != np.arange(num_pedestrians)).any():
            walls_overlaps += wall_overlaps(positions, wallsx, wallsy, r)

        getmyDensity(positions, myDens)
        
        
        plt.clf()
        plt.ylim(-4.2, 6.1)
        plt.xlim(-9, 6.3)

        #for i in range(len(deleted)):
        #    if deleted[i] == -1:
        plt.scatter(positions[:, 0], positions[:, 1], color='blue', label = "SFM pedestrian position")
        plt.scatter(iterations[t, :, 0], iterations[t, :, 1], color='red', label = "Measured pedestrian position")
        #plt.scatter(0, 0, color='white')
        
        plt.scatter(wallsx, wallsy, color='black', s = 5)
        plt.scatter(exit_doorx, exit_doory, color='green', s = 20)
        plt.title(f'Iteration {t+1}')
        plt.legend()
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.pause(0.01)
        
    
    plt.show()
        

    ped_left = still_in(positions, xstart=room_width/2, xend=room_width - 0.2)

    return myDens, total_overlaps+walls_overlaps



def grid_search(iterations):

    # Initial positions and velocities of pedestrians
    #positions = np.random.rand(num_pedestrians, 2) * np.array([room_width/2, room_height])

    A = 0.0  # Interaction strength
    B = 0.1 # Interaction range
    C = 0.0
    D = 0.1



    A = 2.1  # Interaction strength
    B = 0.3 # Interaction range
    C = 10
    D = 0.2

    A = 4.85  # Interaction strength
    B = 0.3 # Interaction range
    C = 17.69
    D = 0.21


    iter0 = copy.deepcopy(iterations[0])

    # Constants
    room_width = 14.4  # meters
    room_height = 6.4  # meters
    door_width = 3
    exit_doorx = 6.3*np.ones(100)
    exit_doory = np.linspace(-2.2, 4.1, 100)
    num_pedestrians = len(iterations[0])
    num_iterations = 1150
    #num_iterations = 50
    dt = 1/16 # Time step
    deleted = -1*np.ones(num_pedestrians)

    # Parameters
    tau = round(0.585 * (5/16),3)  # Relaxation time

    m = 1  # Pedestrian mass
    v0 = 1.34  # Desired velocity 
    #v0 = 1.5551577064433808

    l = 0.61
    l = 1


    # Initial positions and velocities of pedestrians
    positions = iter0

    nw = 1000

    wallsy1 = np.linspace(-2.2, -0.2, nw)
    wallsx1 = np.zeros(nw)

    wallsy2 = np.linspace(2.2, 4.1, nw)
    wallsx2 = np.zeros(nw)

    wallsy3 = np.linspace(-2.2, 4.1, nw)
    wallsx3 = -8*np.ones(nw)

    #wallsy4 = np.linspace(-2.2, 4.1, nw)
    #wallsx4 = 6.3*np.ones(nw)

    wallsy5 = 4.1*np.ones(nw)
    wallsx5 = np.linspace(-8, 6.3, nw)

    wallsy6 = -2.2*np.ones(nw)
    wallsx6 = np.linspace(-8, 6.3, nw)



    wallsy = np.concatenate((wallsy1, wallsy2, wallsy3, wallsy5, wallsy6))

    wallsx = np.concatenate((wallsx1, wallsx2, wallsx3, wallsx5, wallsx6))

    velocities = np.ones((num_pedestrians, 2))*v0 * [1, 0]


    r = 0.2*np.ones((num_pedestrians)) # Pedestrian radius

    gridResol = 3
    A = np.round(np.linspace(25, 30, gridResol),2)
    B = np.round(np.linspace(1.9, 2.3, gridResol),2)

    C = np.round(np.linspace(8, 15, gridResol),2)
    D = np.round(np.linspace(0.3, 1.2, gridResol),2)

    kn = 2
    kt = 2

    minPedOver = float('inf')
    minWallsOver = float('inf')
    minParams = [0, 0, 0, 0]

    top5 = []

    i = 0
    j = 0
    for a in A:
        for b in B:
            for c in C:
                
                for d in D:
                    j+=1
                    if j%4 == 1:
                        suffix = "/"
                    elif j%4 == 2:
                        suffix = "-"
                    elif j%4 == 3:
                        suffix = "\\"
                    else:
                            suffix = "|"
                    print("", end=f"\r{suffix} {room_height, room_width, num_pedestrians} Best parameters found yet : {minParams} | Result : {minPedOver, minWallsOver} | Top5 : {top5} | PercentComplete: {round(i,2)} %")
                    i += 1/gridResol**4 * 100  


                    deleted = -1*np.ones(num_pedestrians)
                    #positions = np.random.rand(num_pedestrians, 2) * np.array([room_width/2, room_height])

                    x0 = np.zeros(num_pedestrians)
                    y0 = np.linspace(0, room_height, num_pedestrians)

                    positions = np.array([[x0[i], y0[i]] for i in range(num_pedestrians)])
                    velocities = np.ones((num_pedestrians, 2))*v0/10
                    total_overlaps, walls_overlaps, ped_left = main_loop(a, b, c, d, kn, kt, room_width, room_height, door_width, exit_doorx,exit_doory, num_pedestrians, num_iterations, dt, deleted, tau, m, v0, l, positions, wallsy, wallsx, velocities, r)
                    if ped_left == 0 and total_overlaps + walls_overlaps < minPedOver + minWallsOver:
                        minPedOver = total_overlaps
                        minWallsOver = walls_overlaps
                        minParams = [a, b, c, d]
                        top5.append(minParams)
                        if len(top5) > 5:
                            top5.pop(0)
                    

    return minParams

#minTau = grid_search(iterations)
#print("\ntau = ", minTau)

A = 0.0  # Interaction strength
B = 0.1 # Interaction range
C = 0.0
D = 0.1


A = 4.85  # Interaction strength
B = 0.3 # Interaction range
C = 17.69
D = 0.21

A = 3.2  # Interaction strength
B = 0.58 # Interaction range
C = 17.5
D = 0.2

A = 2.1  # Interaction strength
B = 0.3 # Interaction range
C = 10
D = 0.2

A = 2.1 # Interaction strength
B = 0.3 # Interaction range
C = 10
D = 0.2


densities = getDensity(iterations)
densities = densities[43::]

print("Density sum :", sum(densities[:-400]))

# Constants
room_width = 14.3  # meters
room_height = 6.3  # meters
door_width = 3
exit_doorx = 6.3*np.ones(100)
exit_doory = np.linspace(-2.2, 4.1, 100)
num_pedestrians = len(iterations[0])
num_iterations = 1150
#num_iterations = 150
dt = 1/16 # Time step



# Parameters
tau = round(0.5 ,3)  # Relaxation time
tau = 0.5
m = 1  # Pedestrian mass
v0 = 1.34  # Desired velocity 
#v0 = 1.5551577064433808

l = 0.61
l = 1
l = 0.5


deleted = -1*np.ones(num_pedestrians)
positions = np.zeros((num_pedestrians,2))
velocities = np.ones((num_pedestrians, 2))*v0*[1e-8,1e-8]

# Initial positions and velocities of pedestrians

nw = 1000

#wallsy1 = np.linspace(-2.2, -0.2, nw)
wallsy1 = np.linspace(-10, -0.2, nw)
wallsx1 = np.zeros(nw)

#wallsy2 = np.linspace(2.2, 4.1, nw)
wallsy2 = np.linspace(2.2, 12, nw)
wallsx2 = np.zeros(nw)

wallsy3 = np.linspace(-2.2, 4.1, nw)
wallsx3 = -8*np.ones(nw)

#wallsy4 = np.linspace(-2.2, 4.1, nw)
#wallsx4 = 6.3*np.ones(nw)

wallsy5 = 4.1*np.ones(nw)
wallsx5 = np.linspace(-8, 6.3, nw)

wallsy6 = -2.2*np.ones(nw)
wallsx6 = np.linspace(-8, 6.3, nw)

wallsy = np.concatenate((wallsy1, wallsy2, wallsy5, wallsy6))
wallsx = np.concatenate((wallsx1, wallsx2, wallsx5, wallsx6))

#wallsy = np.concatenate((wallsy1, wallsy2))
#wallsx = np.concatenate((wallsx1, wallsx2))

r = 0.2*np.ones((num_pedestrians)) # Pedestrian radius

#dens1, over1 = main_loop(A, B, C, D, 2, 2, room_width, room_height, door_width, exit_doorx, exit_doory, num_pedestrians, num_iterations, dt, deleted, tau, m , v0, l, positions, wallsy, wallsx, velocities, r, iterations, Helbing=True)

#err1 = getDensityError(densities, dens1)



deleted = -1*np.ones(num_pedestrians)
positions = np.zeros((num_pedestrians,2))
velocities = np.ones((num_pedestrians, 2))*v0*[1e-8,1e-8]

A = 2.95
B = 0.55
l = 0.1
tau = 0.22  # Relaxation time
dens2, over2 = main_loop(A, B, C, D, 2, 2, room_width, room_height, door_width, exit_doorx, exit_doory, num_pedestrians, num_iterations, dt, deleted, tau, m , v0, l, positions, wallsy, wallsx, velocities, r, iterations)

err2 = getDensityError(densities, dens2)


#print("Overlaps 1:", over1)
#print("DensError :", err1)

print("Overlaps 2:", over2)
print("DensError :", err2)

plt.figure()
plt.title("Density in the concerned area over time")
plt.plot(densities, c = "r", label = "AO data density")
#plt.plot(dens1, c = "g", label = "SFM density, $(A, B, tau)$ = (" + str(2.1) + "," + str(0.3) + "," + str(0.5) + ")")
plt.plot(dens2, c = "b", label = "SFM density, $(A, B, tau, \lambda)$ = (" + str(A) + "," + str(B) + "," + str (tau) + "," + str(l) +")")
plt.xlabel("Iteration")
plt.ylabel("Density $[m^{-2}]$")
plt.grid()
plt.legend()

plt.show()