import numpy as np
import matplotlib.pyplot as plt
import skfmm

def wold(e, f):
    if np.dot(e, f) >= np.linalg.norm(f)*np.cos(np.pi*100/180):
        return 1
    return 0.5

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


# Constants
room_width = 8  # meters
room_height = 15  # meters
door_width = 2.4
exit_doory = np.linspace(room_height/2 - door_width/2 , room_height/2 + door_width/2, 100)
exit_doorx = room_width*np.ones(100)
num_pedestrians = 30
num_iterations = 300
dt = 0.0625  # Time step
deleted = -1*np.ones(num_pedestrians)


# Parameters
tau = 0.5  # Relaxation time
m = 1  # Pedestrian mass
v0 = 1.34  # Desired velocity 

# speed = v0*np.ones_like(X)
# times = skfmm.travel_time(phi, speed, dx=1e-2)

l = 0.61

A = 2.1    # Interaction strength
B = 0.3 # Interaction range
C = 10
D = 0.2

kn = 2
kt = 2

# Initial positions and velocities of pedestrians
positions = np.random.rand(num_pedestrians, 2) * np.array([room_width/2, room_height])

x0 = np.random.randint(0, 2, num_pedestrians)
x0 = np.zeros(num_pedestrians)
y0 = np.linspace(0, room_height, num_pedestrians)

positions = np.array([[x0[i], y0[i]] for i in range(num_pedestrians)])

nw = 1000
wallsy1 = np.linspace(0, room_height/2 - door_width/2, nw//2)
wallsy2 = np.linspace(room_height/2 + door_width/2, room_height, nw//2)
wallsy = np.append(wallsy1, wallsy2)
wallsx = (room_width)*np.ones(nw)


#positions = np.array([find_closest_point(position, grid) for position in positions])

#velocities = np.random.rand(num_pedestrians, 2) * v0

velocities = np.ones((num_pedestrians, 2))*v0/10

r = 0.2*np.ones((num_pedestrians)) # Pedestrian radius

def count_overlaps(positions, xstart, xend):
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

def wall_overlaps(positions, wallsx, wallsy):
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

def social_force_vadere(positions, velocities, deleted):
    forces = np.zeros_like(velocities)

    for i in range(num_pedestrians):
        if deleted[i] != -1 :
                continue

        pos = positions[i]

        #indices = np.where((np.isclose(grid[:, :, 0], pos[0])) & (np.isclose(grid[:, :, 1], pos[1])))
        #idx = [indices[0][0], indices[1][0]]
        
        ed, minD = get_vect(pos, exit_doorx[::], exit_doory[::])
        Fd = (v0*ed - velocities[i])/tau
        forces[i] += Fd

        ew, minW = get_vect(pos, wallsx, wallsy)
        ew = -ew
        Fw =  C* np.exp((r[i]*0 - minW) / D) * ew
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
                b = 0.5*np.sqrt((dij + np.linalg.norm(xdij - velocities[j]*dt))**2 - (np.linalg.norm(velocities[j])*dt)**2)
                Fi =  A* np.exp((-b) / B) * nij * (l + (1-l)*((1+cosphi)/2))
                forces[i] += Fi
                #forces[i] += Ff
        if minD <= 0.1 :
            deleted[i] = i
    return forces

# Function to compute the social force
def social_force(positions, velocities, deleted):
    forces = np.zeros_like(velocities)
    Dt = 2

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

                d2 = np.linalg.norm(xdij - velocities[j]*Dt)
                cosphi = -np.dot(nij, ei)
                b = 0.5*np.sqrt((dij + np.linalg.norm(xdij - velocities[j]*Dt))**2 - (np.linalg.norm(velocities[j])*Dt)**2)

                Fi =  A* np.exp((-b) / B) * (dij + d2)/(2*b) * (xdij/dij + (xdij - velocities[j]*Dt)/d2)/2
                w = wold(ei, -Fi)
                forces[i] += w*Fi

                Dv = velocities[j] - velocities[i]
                tij = np.array([-nij[1], nij[0]])
                Ff = kn*np.heaviside(rij - dij, 0)*nij + kt*np.heaviside(rij - dij, 0)*np.dot(Dv,tij)*tij
                #forces[i] += Ff
        if minD <= 0.1 :
            deleted[i] = i
    return forces

total_overlaps = 0
walls_overlaps = 0
# Simulation loop
for t in range(num_iterations):
    if (deleted == np.arange(num_pedestrians)).all():
        break
    #if t%10 == 0:
    #    print("iteration", t)

    forces = social_force_vadere(positions, velocities, deleted)
    accelerations = forces / m
    velocities += accelerations * dt
    
    for i in range(len(velocities)):
        if deleted[i] != -1:
            velocities[i] = np.array([0,0])
    positions += velocities * dt
    total_overlaps += count_overlaps(positions, xstart=room_width/2, xend=room_width - 0.1)
    if (deleted != np.arange(num_pedestrians)).any():
        
        if t%10 == 0:
            print("iteration", t)
        walls_overlaps += wall_overlaps(positions, wallsx, wallsy)
    #positions = np.array([find_closest_point(position, grid) for position in positions])
    

    # Plot the positions of pedestrians
    
    plt.clf()
    plt.xlim(-1, room_width+1)
    plt.ylim(-1, room_height+1)

    radius = 0.25*10
    s = (np.pi*radius)**2
    plt.scatter(positions[:, 0], positions[:, 1], color='blue')
    
    plt.scatter(wallsx, wallsy, color='black', s = 20)
    plt.scatter(exit_doorx, exit_doory, color='red', s = 20)
    plt.title(f'Iteration {t+1}')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.pause(0.1)
    
#print(deleted == np.arange(num_pedestrians))
ped_left = still_in(positions, xstart=room_width/2, xend=room_width - 0.2)
print("ped overlaps =", total_overlaps)
print("walls overlaps =", walls_overlaps)
print("pedestrians left = ", ped_left)
plt.show()
