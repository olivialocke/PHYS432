import numpy as np
import matplotlib.pyplot as plt
import math
from IPython.display import display, clear_output

## define parameters  
Ngrid = 60 # plot's grid size
Nsteps = 700 # number of steps
dt = 0.01 # timestep size
dx = 1.0 # x-step size 

# define and label index steps for easy future use 
# used to keep track of indices while solving hydro equations 
N = Ngrid
N_ = N+1
N_1 = N
N_2 = N-1
N_3 = N-2

# set up arrays to use in calculations
x = np.arange(0.0, N*dx, dx) # x distance variable
f1 = np.ones(N) # f1 = rho
f2 = np.zeros(N) # f2 = rho * u
f3 = np.ones(N) # f3 = rho* total energy
u = np.zeros(N+1) # u = advective velocity
Jflux = np.zeros(N+1) # array for temporary calculation of flux thru cells

## set up plot
plt.ion()
fig, axes = plt.subplots(2, 1)

# titles of plots
axes[0].set_ylabel('Density (kg/m^3)')
axes[1].set_ylabel('Mach Number')
axes[1].set_xlabel('x (m)')

## calculate a gaussian with (a=amplitude, b=mean, c=stddev)
a=270.0
b=N*dx/2.0
c=5.0

def gaussian(x):
    return a*math.exp(-(x-b)**2.0/c**2.0)

# add gaussian perturbation to f3 (energy)
for xx in range(0, N_1): # for every value 0 to N across the f3 grid
    f3[xx] = f3[xx]*(1.0 + gaussian(xx*dx)) # f3 = 1 + gaussian at time zero

## define important variables
P = (5/3-1) * f1 * (f3/f1 - (f2/f1)**2/2) # pressure 
Cs = np.sqrt(5/3 * P/f1) # speed of sound
Mach = u[0:N] / Cs # Mach number
    
## we will be updating these plotting objects
plt1, = axes[0].plot(x, f1, 'r-') # density
plt2, = axes[1].plot(x, Mach, 'b-') # Mach number

## set axes limits
# x axis 
axes[0].set_xlim([0,N*dx])
axes[1].set_xlim([0,N*dx])

# y axis
axes[0].set_ylim([0.001,7])
axes[1].set_ylim([-2,2])

fig.canvas.draw() # draw plot

## evolution
count = 0 
while count < Nsteps:
    
    ## calculate velocity u at the cell centers, note u[0] = u[N] = 0 
    u[1:N_1] = 0.5*(f2[0:N_2]/f1[0:N_2] + f2[1:N_1]/f1[1:N_1])
    # 1...N-1        0...N-2    0...N-2     1...N-1   1...N-1
 
    ## advect denisty and momentum
    # calculate flux for f1 = rho, note Jflux[0] = Jflux[N] = 0
    for i in range(1, N_1):  # range 1 to N-1
        if (u[i] > 0.0):
            Jflux[i] = u[i]*f1[i-1]
        else:
            Jflux[i] = u[i]*f1[i]
 
    # calculate f1 function; continuity equation; conservation of mass
    f1[0:N_1] = f1[0:N_1] - dt/dx * (Jflux[1:N_] - Jflux[0:N_1])
    # 0...N-1      0...N-1                 1...N         0...N-1

    # calculate flux for f2 = rho*u = momentum
    # same as for f1, replacing f1 for f2
    for i in range(1, N_1):
        if (u[i] > 0.0):
            Jflux[i] = u[i]*f2[i-1]
        else:
            Jflux[i] = u[i]*f2[i]
         
    # calculate f2 function; Euler equation; conservation of momentum
    f2[0:N_1] = f2[0:N_1] - dt/dx * (Jflux[1:N_] - Jflux[0:N_1])
 
    ## compute pressure
    P = (5/3 - 1) * f1 * (f3/f1 - (f2/f1)**2 / 2)
    
    ## calculate source term for momentum (apply the pressure gradient force to the momentum equation)
    f2[1:N_2] = f2[1:N_2] -  (dt/(2*dx)) * (P[2:N_1] - P[0:N_3])
    
    ## correct for the pressure gradient force at the simulation boundaries (reflective boundary conditions)
    f2[0] = f2[0] - 0.5* dt/dx * (P[1] - P[0]) # first cell
    f2[N-1] = f2[N-1] - 0.5* dt/dx * (P[N-1] - P[N-2]) # last cell
    
    ## re-calculate advection velocities 
    u[1:N_1] = 0.5*( f2[0:N_2]/f1[0:N_2] + f2[1:N_1]/f1[1:N_1] )
    
    ##  advect energy  
    # same method as for density and momentum
    for i in range(1, N_1):
        if (u[i] > 0.0):
            Jflux[i] = u[i]*f3[i-1]
        else:
            Jflux[i] = u[i]*f3[i]
 
    # calculate f3 function; energy conservation
    f3[0:N_1] = f3[0:N_1] - dt/dx * (Jflux[1:N_] - Jflux[0:N_1])
    
    ## re-compute pressure as f2, f3 values have changed  
    P = (5/3-1) * f1 * (f3/f1 - (f2/f1)**2/2)
    
    ## apply source term to the energy equation
    f3[1:N_2] = f3[1:N_2] -  (dt/(2*dx)) * (P[2:N_1] * (f2[2:N_1] / f1[2:N_1]) - P[0:N_3] * (f2[0:N_3] / f1[0:N_3]))
    
    ## correct for the source term at the simulation boundaries (reflective boundary conditions)
    f3[0] = f3[0] - 0.5* dt/dx * (P[1]*(f2[1] / f1[1]) - P[0]*(f2[0] / f1[0])) # first cell
    f3[N-1] = f3[N-1] - 0.5* dt/dx * (P[N-1]*(f2[N-1] / f1[N-1]) - P[N-2]*(f2[N-2] / f1[N-2])) # last cell
    
    ## re-calculate pressure as f3 values have been changed
    P = (5/3-1) * f1 * (f3/f1 - (f2/f1)**2/2)

    ## update sound speed and Mach number
    Cs = np.sqrt(5/3 * P/f1)
    Mach = u[0:N] / Cs

    ## update the plots
    plt1.set_ydata(f1)
    plt2.set_ydata(Mach)
    
    # display plots
    display(fig)
    clear_output(wait=True)
    
    fig.canvas.draw()
    plt.pause(0.001)
    
    # add to count (for next loop through)
    count += 1

