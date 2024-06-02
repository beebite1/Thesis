#!/bin/python3

import numpy as np
#import matplotlib.pyplot as plt
import copy
from fteikpy import Eikonal2D
import sys

arg1 = int(sys.argv[1])
arg2 = int(sys.argv[2])
arg3 = int(sys.argv[3])
arg4 = int(sys.argv[4])


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
    for i in range(50,len(myDens)-400):
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
    vect = res/(np.linalg.norm(res)+1e-8)
    return vect, minD

def is_in_array(position, deleted):
    for elem in deleted:
        compx = position[0] == elem[0]
        compy = position[1] == elem[1]
        if compx and compy:
            return True
        if np.equal(position, elem).all():
            #print(np.equal(position, elem).all())
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

def social_force_vadere(A, B, C, D,  kn, kt, room_width, room_height, door_width, exit_doorx,
              exit_doory, 
              num_pedestrians, num_iterations, dt, deleted,
              tau, m, v0, l, positions, wallsy, wallsx,
              velocities, r, tt):
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
                #ej = velocities[j]/np.linalg.norm(velocities[j])
                d2 = np.linalg.norm(xdij - velocities[j]*Dt)
                cosphi = -np.dot(nij, ei)
                b = 0.5*np.sqrt((dij + np.linalg.norm(xdij - velocities[j]*Dt))**2 - (np.linalg.norm(velocities[j])*Dt)**2)
                Fi =  A* np.exp((rij*0 -b) / B) * (l + (1-l)*((1+cosphi)/2)) * (dij + d2)/(2*b) * (xdij/dij + (xdij - velocities[j]*Dt)/d2)/2
                forces[i] += Fi

        if minD <= 0.1 :
            deleted[i] = i
    return forces

def realDist(positions, iterations, deleted):
    res = 0
    for i in range(len(deleted)):
        if deleted[i] == -1:
            res += np.sqrt((positions[i,0]-iterations[i,0])**2 + (positions[i,1]-iterations[i,1])**2)
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
    tt = floorfield(room_width, room_height)

    myDens = []
    # Simulation loop
    for t in range(num_iterations):
        if t < 43:
            continue

        for pedId in range(num_pedestrians):
            if deleted[pedId] == pedId and (iterations[t, pedId, 0] != 0 or iterations[t, pedId, 1] != 0):
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
              velocities, r, tt)

        # F = ma (and motion equations)
        accelerations = forces / m
        velocities += accelerations * dt
        
        # If pedestrian arrived, he doesn't move anymore
        for i in range(len(velocities)):
            if deleted[i] != -1:
                velocities[i] = np.array([1e-8,1e-8])
        positions += velocities * dt

        # Count metrics
        #rDist = realDist(positions, iterations[t])
        #distanceNorm += np.linalg.norm(rDist)

        total_overlaps += count_overlaps(positions, r, deleted)
        if (deleted != np.arange(num_pedestrians)).any():
            walls_overlaps += wall_overlaps(positions, wallsx, wallsy, r)

        getmyDensity(positions, myDens)

    ped_left = still_in(positions, xstart=-8, xend=6)

    return total_overlaps, walls_overlaps, ped_left, myDens



def grid_search(iterations):

    densities = getDensity(iterations)
    densities = densities[43::]

    # Constants
    room_width = 14.3  # meters
    room_height = 6.3  # meters
    door_width = 2.4
    exit_doorx = 6.3*np.ones(100)
    exit_doory = np.linspace(-2.2, 4.1, 100)
    num_pedestrians = len(iterations[0])
    num_iterations = 1150
    dt = 1/16 # Time step
    deleted = -1*np.ones(num_pedestrians)

    # Parameters
    tau = 0.22  # Relaxation time

    m = 1  # Pedestrian mass
    v0 = 1.34  # Desired velocity 
    l = 1

    # Initial positions and velocities of pedestrians
    nw = 1000

    wallsy1 = np.linspace(-10, -0.2, nw)
    wallsx1 = np.zeros(nw)
    
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


    r = 0.2*np.ones((num_pedestrians)) # Pedestrian radius

    gridResol = 5
    A = np.round(np.linspace(2, 6, gridResol),2)
    A = [A[arg1]]
    B = np.round(np.linspace(0.1, 0.5, gridResol),2)
    B = [B[arg2]]
    C = np.round(np.linspace(10, 50, gridResol),2)
    C = np.array([1, 10, 20, 30, 50])
    C = [C[arg3]]
    D = np.round(np.linspace(0.1, 1, gridResol),2)
    D = np.array([0.2, 0.4, 0.6, 0.8, 1])
    D = [D[arg4]]

    kn = 2
    kt = 2

    minPedOver = float('inf')
    minWallsOver = float('inf')
    minError = float('inf')
    minParams = [0, 0, 0, 0]
    err = float('inf')

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
                    #print("", end=f"\r{suffix} Current : {(a, b, c, d)} Result : {round(err,2)} | Best : {minParams} Result : {round(minError,2)} | Top5 : {top5} | Completed: {round(i,2)} %")
                    i += 1/gridResol**4 * 100  


                    deleted = -1*np.ones(num_pedestrians)
                    positions = np.zeros((num_pedestrians,2))
                    velocities = np.ones((num_pedestrians, 2))*v0*[1,1]*1e-8

                    total_overlaps, walls_overlaps, ped_left, dens = main_loop(a, b, c, d, kn, kt, room_width, room_height, door_width, exit_doorx,exit_doory, num_pedestrians, num_iterations, dt, deleted, tau, m, v0, l, positions, wallsy, wallsx, velocities, r, iterations)
                    err = getDensityError(densities, dens)

                    #if ped_left == 0 and total_overlaps + walls_overlaps < minPedOver + minWallsOver:
                    if err**2 + total_overlaps + walls_overlaps < minError**2 + minPedOver + minWallsOver:
                        minPedOver = total_overlaps
                        minWallsOver = walls_overlaps
                        minError = err
                        minParams = [a, b, c, d]
                        top5.append(minParams)
                        if len(top5) > 5:
                            top5.pop(0)
                    

    return top5, round(minError, 2), minPedOver, minWallsOver

top5, minErr, minPed, minWalls = grid_search(iterations)
print("\nresults = ", top5, minErr, minPed, minWalls)