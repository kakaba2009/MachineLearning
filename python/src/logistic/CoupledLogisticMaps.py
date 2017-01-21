# CoupledLogisticMaps.py: Plot out iterates of two coupled logistic maps
#
# Two coupled logistic maps are given by
#     x_n+1 = f( x_n , y_n )
#     y_n+1 = g( x_n , y_n )
# with
#     f( x_n , y_n ) = r * x_n (1.0 - x_n) + c * y_n
#     g( x_n , y_n ) = r * y_n (1.0 - y_n) + c * x_n
#
# The state space is R^2
# and the control parameters range in
#       r in [0,4]
#       c in [0,1]
# 

def CoupledLogisticMaps(r,c,x,y):
	return r * x * (1.0 - x) + c * y, r * y * (1.0 - y) + c * x

# Import plotting routines
from pylab import *

# Simulation parameters
#
# Control parameters:
r = 3.5
c = 0.05
# The number of iterations to throw away
nTransients = 100
# The number of iterations to generate
nIterates = 10000

# Initial condition
xtemp = 0.1
ytemp = 0.3
for n in range(0,nTransients):
  xtemp, ytemp = CoupledLogisticMaps(r,c,xtemp,ytemp)
# Set up arrays of iterates (x_n,y_n) and set the initital condition
x = [xtemp]
y = [ytemp]
# The main loop that generates iterates and stores them
for n in range(0,nIterates):
  # at each iteration calculate (x_n+1,y_n+1)
  xtemp, ytemp = CoupledLogisticMaps(r,c,x[n],y[n])
  # and append to lists x and y
  x.append( xtemp )
  y.append( ytemp )

# Setup the plot
xlabel('x(n)') # set x-axis label
ylabel('y(n)') # set y-axis label
title('Coupled Logistic Maps with (r,c) = (' + str(r) + ',' + str(c) + ')') # set plot title
# Plot the time series
plot(x,y, 'r,')
# Make sure the iterates appear in a unit square
axis('equal')
axis([0.0,1.0,0.0,1.0])

# Use command below to save figure
#savefig('HenonMapIterates', dpi=600)

# Display the plot in a window
show()
