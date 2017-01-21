# LogisticXMax.py:
#   What are those "veils" in the bifurcation diagram?
#	Overlay iterates of the map's maximum on a Logistic bifurcation diagram.

# Import modules
from numpy import *
from pylab import *

# Define the Logistic map's function 
def LogisticMap(r,x):
	return r * x * (1.0 - x)

# Setup parameter range
rlow  = 2.9
rhigh = 4.0

# Setup the plot
# It's numbered 1 and is 8 x 6 inches.
figure(1,(8,8))
# Stuff parameter range into a string via the string formating commands.
TitleString = 'Logistic map for r in [%g,%g]: Attractor and Iterates of the Maximum ' % (rlow,rhigh)
title(TitleString)
# Label axes
xlabel('Control parameter r')
ylabel('{X(n)}')

# Set the initial condition used across the different parameters
ic = 0.2
# Establish the arrays to hold the set of iterates at each parameter value
rsweep = [ ] # The parameter value
x = [ ] # The iterates
# The iterates we'll throw away
nTransients = 500
# This sets how much the attractor is filled in
nIterates = 250
# This sets how dense the bifurcation diagram will be
nSteps = 500.0

# First, the bifurcation diagram
# Sweep the control parameter over the desired range
for r in arange(rlow,rhigh,(rhigh-rlow)/nSteps):
	# Set the initial condition to the reference value
	state = ic
	# Throw away the transient iterations
	for i in range(nTransients):
		state = LogisticMap(r,state)
	# Now store the next batch of iterates
	for i in range(nIterates):
		state = LogisticMap(r,state)
		rsweep.append(r)
		x.append( state )

# Plot the list of (r,x) pairs as pixels
plot(rsweep, x, 'b,')

# Now for the iterates of the logistic map's maximum
# The map's maximum is at 1/2.
ic = 0.5
nItsMax = 6
# Sweep the control parameter over the desired range
for nIts in range(nItsMax):
	# Establish the arrays and clear them
	rsweep = []
	x = []
	for r in arange(rlow,rhigh,(rhigh-rlow)/nSteps):
		rsweep.append(r)
		# Set the initial condition to the reference value
		state = ic
		# Store the nIts iterate of the map; we iterate at least once
		for i in range(nIts+1):
			state = LogisticMap(r,state)
		x.append( state )
	plot(rsweep, x, 'r,')

# Use this to save figure as a bitmap png file
#savefig('LogisticXMax', dpi=600)

# Display plot in window
show()
