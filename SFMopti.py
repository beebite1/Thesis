import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def get_vect(position, exit_doorx, exit_doory):
    distances = np.sqrt((exit_doorx - position[0])**2 + (exit_doory - position[1])**2)
    minDidx = np.argmin(distances)
    minD = distances[minDidx]

    dx = exit_doorx[minDidx] - position[0]
    dy = exit_doory[minDidx] - position[1]

    res = np.array([dx, dy])
    vect = res/np.linalg.norm(res)
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

    for i in range(num_pedestrians):
        if deleted[i] != -1 :
                continue

        pos = positions[i]
        
        ed, minD = get_vect(pos, exit_doorx, exit_doory)
        Fd = (v0*ed - velocities[i])/tau
        forces[i] += Fd

        ew, minW = get_vect(pos, wallsx, wallsy)
        ew = -ew
        Fw =  C* np.exp((r[i]*0 - minW) / D) * ew
        forces[i] += Fw

        #tiw = np.array([-ew[1], ew[0]])
        #vi = velocities[i]
        #Ffw = kn*np.heaviside(r[i] - minW, 0)*ew - kt*np.heaviside(r[i] - minW, 0)*np.dot(vi, tiw)*tiw

        #forces[i] += Ffw

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

                #Dv = velocities[j] - velocities[i]
                #tij = np.array([-nij[1], nij[0]])
                #Ff = kn*np.heaviside(rij - dij, 0)*nij + kt*np.heaviside(rij - dij, 0)*np.dot(Dv,tij)*tij
                #forces[i] += Ff
        if minD <= 0.1 :
            deleted[i] = i
    return forces

# Function to compute the social force
def social_force(A, B, C, D,  kn, kt, room_width, room_height, door_width, exit_doorx,
              exit_doory, 
              num_pedestrians, num_iterations, dt, deleted,
              tau, m, v0, l, positions, wallsy, wallsx,
              velocities, r):
    forces = np.zeros_like(velocities)

    for i in range(num_pedestrians):
        if deleted[i] != -1 :
                continue

        pos = positions[i]

        #indices = np.where((np.isclose(grid[:, :, 0], pos[0])) & (np.isclose(grid[:, :, 1], pos[1])))
        #idx = [indices[0][0], indices[1][0]]
        
        ed, minD = get_vect(pos, exit_doorx, exit_doory)
        Fd = (v0*ed - velocities[i])/tau
        forces[i] += Fd

        ew, minW = get_vect(pos, wallsx, wallsy)
        ew = -ew
        Fw =  C* np.exp((r[i] - minW) / D) * ew
        forces[i] += Fw

        tiw = np.array([-ew[1], ew[0]])
        vi = velocities[i]
        Ffw = kn*np.heaviside(r[i] - minW, 0)*ew - kt*np.heaviside(r[i] - minW, 0)*np.dot(vi, tiw)*tiw

        #forces[i] += Ffw

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

                cosphi = -np.dot(nij, ei)
                Fi =  A* np.exp((rij - dij) / B) * nij * (l + (1-l)*((1+cosphi)/2))
                #Fi = A* np.exp((rij - dij) / B) * nij
                forces[i] += Fi

                Dv = velocities[j] - velocities[i]
                tij = np.array([-nij[1], nij[0]])
                Ff = kn*np.heaviside(rij - dij, 0)*nij + kt*np.heaviside(rij - dij, 0)*np.dot(Dv,tij)*tij
                #forces[i] += Ff
        if minD <= 0.1 :
            deleted[i] = i
    return forces

def main_loop(A, B, C, D, kn, kt, room_width, room_height, door_width, exit_doorx,
              exit_doory, 
              num_pedestrians, num_iterations, dt, deleted,
              tau, m, v0, l, positions, wallsy, wallsx,
              velocities, r, model = social_force_vadere):
    total_overlaps = 0
    walls_overlaps = 0
    # Simulation loop
    for t in range(num_iterations):
        #if t%10 == 0:
        #    print("iteration", t)
        if (deleted == np.arange(num_pedestrians)).all():
            break

        forces = model(A, B, C, D, kn, kt, room_width, room_height, door_width, exit_doorx,
              exit_doory, 
              num_pedestrians, num_iterations, dt, deleted,
              tau, m, v0, l, positions, wallsy, wallsx,
              velocities, r)
        accelerations = forces / m
        velocities += accelerations * dt
        
        for i in range(len(velocities)):
            if deleted[i] != -1:
                velocities[i] = np.array([0,0])
        positions += velocities * dt

        total_overlaps += count_overlaps(positions, room_width/2, room_width - 0.1, r)
        if (deleted != np.arange(num_pedestrians)).any():
            walls_overlaps += wall_overlaps(positions, wallsx, wallsy, r)

    ped_left = still_in(positions, xstart=room_width/2, xend=room_width - 0.2)

    return total_overlaps, walls_overlaps, ped_left



def grid_search():
    # Constants
    room_width = 8  # meters
    room_height = 15  # meters
    door_width = 2.4
    exit_doory = np.linspace(room_height/2 - door_width/2 , room_height/2 + door_width/2, 100)
    exit_doorx = room_width*np.ones(100)
    num_pedestrians = 25
    num_iterations = 160
    dt = 0.2  # Time step
    dt = 1/16
    deleted = -1*np.ones(num_pedestrians)

    # Parameters
    tau = 0.5 * (5/16)  # Relaxation time
    m = 1  # Pedestrian mass
    v0 = 1.34  # Desired velocity 

    l = 0.61

    # Initial positions and velocities of pedestrians
    positions = np.random.rand(num_pedestrians, 2) * np.array([room_width/2, room_height])

    nw = 1000
    wallsy1 = np.linspace(0, room_height/2 - door_width/2, nw//2)
    wallsy2 = np.linspace(room_height/2 + door_width/2, room_height, nw//2)
    wallsy = np.append(wallsy1, wallsy2)
    wallsx = (room_width)*np.ones(nw)

    velocities = np.ones((num_pedestrians, 2))*v0

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

    #total_overlaps = 10000
    #walls_overlaps = 10000
    #ped_left = 10000

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

minParams = grid_search()

#main_loop(0.1, 0.05, 0.1, 0.05, 2, 2)

print("\na, b, c, d = ", minParams)