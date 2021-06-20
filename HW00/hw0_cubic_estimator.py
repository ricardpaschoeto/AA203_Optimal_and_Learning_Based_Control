import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def cubic(x, c0, c1, c2, c3):
    return c3 + c2*x + c1*pow(x,2) + c0*pow(x,3)

def read_data():
    diameter = []
    mass =[]
    for line in open('prob3data.csv','r'):
        data = line.split(',')
        diameter.append(float(data[0]))
        mass.append(float(data[1]))
    return [diameter, mass]

diameter, mass = read_data()
popt, pcov = curve_fit(cubic, diameter, mass)
curve_fitted = cubic(np.array(diameter), *popt)
print('{} {} {} {}'.format(popt[0], popt[1],popt[2], popt[3]))
fig, ax = plt.subplots()
ax.plot(diameter, mass, color='blue')
ax.plot(diameter, curve_fitted, 'r-')
ax.grid()
plt.show()