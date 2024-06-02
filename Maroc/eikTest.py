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

def floorfield(positions, room_height, room_width):
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

        wallval = 1e-3
        velocity_model = np.ones((m, n))
        velocity_model[0,:] = wallval
        velocity_model[:,n-1] = wallval
        velocity_model[int(3/resol):,:int(4/resol)] = wallval
        #velocity_model[3:,3] = wallval

        dx, dy = resol, resol

        # Solve Eikonal at source
        eik = Eikonal2D(velocity_model, gridsize=(dx, dy))
        tt = eik.solve((7, 5), return_gradient=True)

        # Get traveltime at specific grid point
        t1 = tt[x, y]

        sz = 50
        X,Y = np.meshgrid(np.linspace(0,7,sz),np.linspace(0,7,sz))

        print(np.shape(X))
        U = np.zeros((sz,sz))
        V = np.zeros((sz,sz))
        t2 = np.zeros((700,700))
        tgrad = np.zeros((700, 700))
        for k in range(700):
            xi = k
            for l in range(700):
                yi = l
                t2[k,l] = tt[xi, yi] 
                tgrad[k,l] = tt.gradient[1][xi][yi]
                if k%14==0 and l%14==0:
                    U[k//14,l//14] = tt.gradient[0][m-xi][yi]
                    V[k//14,l//14] = -tt.gradient[1][m-xi][yi]

        """


        plt.imshow(tgrad, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()


        """
        plt.figure()
        plt.title("Eikonal equation resolution : Initial conditions on the domain")
        plt.imshow(velocity_model, cmap='YlGnBu', interpolation='nearest')
        plt.xlabel("X[cm]")
        plt.ylabel("Y[cm]")
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.title("Eikonal equation resolution : Remaining travel time")
        plt.imshow(t2, cmap='YlGnBu', interpolation='nearest', vmin=0, vmax=10)
        plt.xlabel("X[cm]")
        plt.ylabel("Y[cm]")
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.title("Eikonal equation resolution : Vector field of the target direction")
        plt.quiver(X,Y,V,U, scale=100)
        plt.xlabel("X[m]")
        plt.ylabel("Y[m]")
        plt.show()

        tgradx = tt.gradient[0][m-y][x]
        tgrady = tt.gradient[1][m-y][x]

        grad = [-tgrady, tgradx]
        allGrads[i] = grad
        i+=1

    return allGrads

positions = [[0,5]]
print(floorfield(positions, 7, 7))
