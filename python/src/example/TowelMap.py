# TowelMap.py: Rossler's Towel Map in interactive 3D.

from enthought.mayavi.mlab import axes,points3d  # @UnresolvedImport
from numpy import arange

print( """
	Rotate: Left-click & drag
	Zoom:   Right-Click & slide up/down
	Slide:  Middle-Click & drag
""")

# Rossler's Towel Map, which is pretty close to two coupled
#    Logistic maps, except in three dimensions
def ThreeDMap(a,b,x):
	x[0] = a * x[0] * (1. - x[0]) - .05 * (x[1] + .35) * (1. - 2. * x[2])
	x[1] = .1 * ( (x[1] + .35) * (1. - 2. * x[2]) - 1.) * (1. - 1.9 * x[0])
	x[2] = b * x[2] * (1. - x[2]) + .2 * x[1]
	return x

  # Towel map parameters
a = 3.9
b = 3.78
# Iterations to throw away
nTransients = 100
# Iterations to plot
nIterates = 8000

  # Initial conditions
x = [0.085,-0.121,0.075]
  # Store the trajectory here for the plotting function
TrajX = []
TrajY = []
TrajZ = []

  # Interate the Towel Map
  #  Throw away transients
  #
for n in range(nTransients):
	x = ThreeDMap(a,b,x)
  #  Collect points
  #
for n in range(nIterates):
	x = ThreeDMap(a,b,x)
	TrajX.append(x[0])
	TrajY.append(x[1])
	TrajZ.append(x[2])

	# Draw a sphere at each new point
	#
points3d(TrajX,TrajY,TrajZ,scale_factor=0.005,color=(1.,0.,0.))
axes(color=(0.,0.,0.),extent=[0.0,1.0,-0.15,0.15,0.0,1.0],
	ranges=[0.0,1.0,-0.10,0.10,0.0,1.0])
