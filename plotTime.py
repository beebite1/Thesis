from matplotlib import pyplot as plt
import numpy as np
import scipy
import scipy.interpolate

SMALL_SIZE = 12
MEDIUM_SIZE = 18
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

prctDirectives = [0, 31, 46, 69, 100]
meanTimes = [254.93258140540544, 350.5520587608696, 333.8575721590909, 232.5007025473684, 236.3873194347826]
stds = np.array([22.510780986343114, 29.840626496695652, 36.30569143137059, 21.642196797098162, 22.55425207768897])


plt.figure()
plt.title("Evolution of the Mean Average Escape Time when Increasing the Amount of Directives")
plt.errorbar(prctDirectives, meanTimes, stds, linestyle="None", marker = "^", capsize=3, ecolor='black', label = "Mean average times and their standard deviation")
plt.plot(prctDirectives, meanTimes,  linestyle = "--")
plt.xlabel("Ratio of gates predetermined [%]")
plt.ylabel("Evacuation time [s]")
plt.legend()
plt.grid(True)
plt.show()