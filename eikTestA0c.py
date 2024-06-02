import numpy as np
from fteikpy import Eikonal2D
import matplotlib.pyplot as plt

SMALL_SIZE = 10
MEDIUM_SIZE = 18
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def floorfield(positions, room_width, room_height):
    numPed = len(positions)
    allGrads = np.zeros((numPed,2))
    i = 0
    for position in positions:
        resol = 1e-2
        x = position[0]
        y = position[1]

        x = int((position[0])/resol)
        y = int((position[1])/resol)

        # Velocity model


        n = int(room_height/resol)
        m = int(room_width/resol)

        wallval = 1e-2
        velocity_model = np.ones((m, n))
        #velocity_model[0,:] = wallval
        #velocity_model[:,n-1] = wallval
        velocity_model[int(8.5/resol):int(9.5/resol),:int(2.1/resol)] = wallval
        velocity_model[int(8.5/resol):int(9.5/resol),int(4.5/resol):] = wallval
        #velocity_model[3:,3] = wallval

        dx, dy = resol, resol

        # Solve Eikonal at source
        eik = Eikonal2D(velocity_model, gridsize=(dx, dy))
        tt = eik.solve((15, 4), return_gradient=True)


        sx = m//10
        sy = n//10
        X,Y = np.meshgrid(np.linspace(0,room_width,sx),np.linspace(0,room_height,sy))
        #Y,X = np.meshgrid(np.linspace(0,room_width,sx),np.linspace(0,room_height,sy))

        U = np.zeros((sy,sx))
        V = np.zeros((sy,sx))
        print(m)
        t2 = np.zeros((m,n))
        for k in range(m):
            xi = k
            for l in range(n):
                yi = l
                t2[k,l] = tt[xi, yi] 
                if k%10==0 and l%10==0:
                    U[l//10,k//10] = -tt.gradient[0][xi][yi]
                    V[l//10,k//10] = -tt.gradient[1][xi][yi]

        """


        plt.imshow(tgrad, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()
        
        
        
        plt.figure()
        plt.title("Eikonal equation resolution : Initial conditions on the domain")
        plt.imshow(velocity_model.T, cmap='YlGnBu', interpolation='nearest')
        plt.xlabel("X[cm]")
        plt.ylabel("Y[cm]")
        plt.colorbar()
        plt.show()
        plt.figure()
        plt.title("Eikonal equation resolution : Remaining travel time")
        plt.imshow(t2.T, cmap='YlGnBu', interpolation='nearest')
        plt.xlabel("X[cm]")
        plt.ylabel("Y[cm]")
        plt.colorbar()
        plt.show()
        """

        plt.figure()
        plt.title("Eikonal equation resolution : Vector field of the target direction")
        plt.quiver(X,Y,U,V, scale=200)
        plt.xlabel("X[m]")
        plt.ylabel("Y[m]")
        plt.show()

        tgradx = tt.gradient[0][x][y]
        tgrady = tt.gradient[1][x][y]

        grad = [-tgradx, -tgrady]
        allGrads[i] = grad
        i+=1

    return allGrads

positions = [[0,5]]
print(floorfield(positions, 15, 7))
