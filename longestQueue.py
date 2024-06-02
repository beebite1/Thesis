import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.interpolate import interp1d

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Directory containing the files
directory = 'final/output'

directory = "C:/Users/brieu/Documents/MASTER/Thesis/Resultats/FinalResults"

def getValues(scenario, id):
    # Variable to store the sum
    values = []
    sum_last_value = 0
    numfiles = 0
    numStuck = 0
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        # Check if the file name starts with "security_Free1234"
        file = "security_" + scenario
        if filename.startswith(file):
            file_path = os.path.join(directory, filename)
            file_path += '/time.txt'
            
            # Open the file
            with open(file_path) as file:
                # Read the second line
                lines = file.readlines()
                if len(lines) >= 2:
                    second_line = lines[1].split()
                    # Extract and sum up the last value of the second line
                    try:
                        last_value = float(second_line[2])
                        sum_last_value += last_value
                        values.append(last_value)

                        if np.abs(last_value - 226.8) < 1:
                            print(last_value)
                            print(filename)
                        numfiles += 1
                    except (ValueError, IndexError):
                        numStuck += 1
                        #print(f"Invalid value in file {filename}")

    values = np.array(values)
    # Print the sum
    print("---------------------------------------------------------------------")
    print(f"Number of stuck simul : {numStuck} || Number of ok simul : {numfiles}")
    #print("Sum of values:", sum(values))
    print("Mean value =", np.mean(values))
    print("Median value =", np.median(values))
    print("Std values =", np.std(values))
    print("---------------------------------------------------------------------")

    """
    # Plot histogram
    plt.hist(values, bins=int(np.sqrt(len(values)))+1,
            density=False, alpha=0.6, color='g', edgecolor = "black",
            label="Measurable sample size = "+ str(numfiles))

    # Add labels and title
    plt.xlabel('Time [s]')
    plt.ylabel('Count')
    plt.title('Histogram of Maximum Escape Time in Scenario '+ str(id))
    plt.grid(True)
    plt.legend()
    # Show plot
    #plt.show()
    """
    return values

allValls = []
scenarios = ["0000", "0200", "0230", "0234", "1234"]
ids = [5, 4, 3, 2, 1]

scenarios = ["1234", "0234", "0230", "0200", "0000"]
ids = [1, 2, 3, 4, 5]

for i in range(len(scenarios)):
    scenario = scenarios[i]
    id = ids[i]
    vals = getValues(scenario, id)
    allValls.append(vals)

ids = [1, 2, 3, 4, 5]
#ids = [5,4,3,2,1]
prct = [0, 31, 46, 69, 100]
#prct =[100, 69, 46, 31, 0]
prctDirectives = np.array(prct)
#meanTimes = [254.93258140540544, 350.5520587608696, 333.8575721590909, 232.5007025473684, 236.3873194347826]
meanTimes = [np.mean(val) for val in allValls]
#stds = np.array([22.510780986343114, 29.840626496695652, 36.30569143137059, 21.642196797098162, 22.55425207768897])
stds = [np.std(val, ddof= 1) for val in allValls]

errors = [2.576*stds[i]/np.sqrt(len(allValls[i])) for i in range(len(allValls))]
#allValls = np.array(allValls)

peakArr = 0.78
Area = 1012
securityStandard = 1.8
depart = 0.72

arrivals = np.zeros(len(allValls))
for i in range(len(allValls)):
    arrivals[i] = Area/(securityStandard*meanTimes[i])+depart

plt.figure()
plt.title("Maximum recommended arrival rate in each Scenario")
plt.scatter(prctDirectives, arrivals, linestyle="None", marker = "D", s= 100, label = "Average maximum recommended arrival rate")
plt.scatter(prctDirectives, peakArr*np.ones(len(allValls)), label = "Current arrival rate at peak half-hour", s= 100)
plt.plot(prctDirectives, arrivals, color = 'blue')
plt.plot(prctDirectives, peakArr*np.ones(len(allValls)), color = 'orange')
plt.xlabel("Ratio of gates predetermined [%]")
plt.ylabel("Arrival rate [pax/s]")
plt.legend()
plt.grid(True)
plt.show()