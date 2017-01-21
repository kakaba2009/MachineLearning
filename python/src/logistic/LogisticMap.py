# LogisticMap.py: Plot out a time series of iterates
#
# The logistic map is given by
#     x_n+1 = f( x_n )
# with
#     f( x_n ) = r * x_n * ( 1 - x_n )
#
# The state space is the unit interval:
#       x in [0,1]
# and the control parameter ranges in
#       r in [0,4]

# Import plotting routines
from pylab import *

# Simulation parameters
#
# Control parameter of the map: A period-3 cycle
r = 3.83
# Set up an array of iterates and set the initital condition
x = [0.1]
# The number of iterations to generate
N = 100

# The main loop that generates iterates and stores them
for n in range(0,N):
  # at each iteration calculate x_n+1
  # and append to list x
  x.append( r*x[n]*(1.-x[n]) )

# Setup the plot
xlabel('Time step n') # set x-axis label
ylabel('x(n)') # set y-axis label
title('Logistic map at r= ' + str(r)) # set plot title
# Plot the time series: once with circles, once with lines
plot(x, 'ro', x , 'b')

# Use command below to save figure
#savefig('LogisticMap', dpi=600)

# Display the plot in a window
show()
