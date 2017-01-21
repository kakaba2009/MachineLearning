# LogisticHistogram.py:
#	Plot a histogram of Logistic map iterates on the interval.

# Import modules
from numpy import *
from pylab import * # plotting

# Define the Logistic map's function 
def LogisticMap(r,x):
	return r * x * (1.0 - x)

# Parameters: Select one by uncommenting it, leaving others commented out
# Here there is a single band
#r = 4.0
# We set r to where two bands merge to one
#r = 3.6785735104283219
# Compare this to a typical chaotic paramter value, one that's nearby
#r = 3.7
# Here is the onset of chaos
#r = 3.5699456718695445
# This is just before the P-3 window appears: Confluence of veils
r = 3.828

# The domain in x that we'll look at
xlow = 0.0
xhigh = 1.0
# The number of histogram bins over that domain
nBins = 1000.0
# The bin size
xInc  = (xhigh-xlow)/nBins
# Setup histogram arrays
# Note how we make sure x = 1.0 is included
x = arange(xlow,xhigh+xInc,xInc)
# This is the array in which we store the bin counts
p = zeros((int(nBins)+1),float)

# Setup the plot
# It's numbered 1 and is 8 x 6 inches.
# On my display this didn't turn out to be square, so I've fudged.
figure(1,(8,6))
TitleString = 'Logistic map histogram at r = %g' % r
title(TitleString)
xlabel('x')       # set x-axis label
ylabel('Prob(x)') # set y-axis label

# Set the initial condition
state = 0.2
# Iterate to throw away
nTransients = 200
# Iterates over which to accumulate histogram counts
# Lots!
nIterates = 2000000
# Let's get started
for n in range(nTransients):
	state = LogisticMap(r,state)
for n in range(nIterates):
	state = LogisticMap(r,state)
	index = int((nBins+1) * state)
	p[index] = p[index] + 1

# Normalize the histogram
p = p / float(nIterates)
# Plot the histogram
plot(x, p, 'b-', antialiased=True)

# Use this to save figure as a bitmap png file
#savefig('LogisticHistogram', dpi=600)

# Display plot in window
show()
