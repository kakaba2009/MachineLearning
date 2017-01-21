# LogisticFunction.py: Plot the Logistic Map's quadratic function

# Import modules
from numpy import *
from pylab import *

# Define the Logistic map's function 
def LogisticMap(r,x):
	return r * x * (1.0 - x)
	#return r * sin(pi*x)

# Setup the x array on the unit interval
# Note how we make sure x = 1.0 is included
x = arange(0.0,1.01,0.01)
# We set r = 3.7, a typical chaotic value
r = 3.7

# Setup the plot
# It's numbered 1 and is 6 x 6 inches, to make the plot square
# On my display this didn't turn out to be square, so I've fudged.
figure(1,(8,8))
# Note how we turn the parameter value into a string
TitleString = 'Logistic map, f(x) = %g x (1 - x), and its second iterate' % r
title(TitleString)
# If LaTeX backend is being used, for genuine typeset math
#xlabel('$X_n$')   # set x-axis label
#ylabel('$X_n+1$') # set y-axis label
# Otherwise
xlabel('X(n)')   # set x-axis label
ylabel('X(n+1)') # set y-axis label

# We plot the identity y = x for reference
plot(x, x, 'b--', antialiased=True)
# The anti-aliasing smoothes otherwise jagged lines

# Here's the Logistic Map itself
plot(x, LogisticMap(r,x), 'g', antialiased=True)

# ... and its second iterate
plot(x, LogisticMap(r,LogisticMap(r,x)), 'r', antialiased=True)

# ... and its third iterate
plot(x, LogisticMap(r,LogisticMap(r,LogisticMap(r,x))), 'y', antialiased=True)

# Use this to save figure as a bitmap png file
#savefig('LogisticFunction', dpi=600)

# Display plot in window
show()
