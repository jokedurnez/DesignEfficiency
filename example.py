from __future__ import division
import numpy as np
import palettable.colorbrewer as cb
import matplotlib.pyplot as plt

from design import design

ISI = 2
TR = 2
L = 255
p = np.array([1/3,1/3,1/3])
C = np.array([
    [ 1, -1,  0],
    [ 1,  0, -1],
    [ 0,  1, -1]
])
rho = 0.3 #temporal autocorrelation, assume AR(1)
weights = np.array([1/4,1/4,1/4,1/4])

reload(design)
des = design.Design(ISI,TR,L,p,C,rho,weights)
order = des.RandomOrder(seed=1)
order = des.CreateDesignMatrix(order)
order = des.ComputeEfficiency(order)
Generation = des.GeneticAlgoritmInitiate()
Generation = des.GeneticAlgoritmCrossover(Generation)
Generation = des.GeneticAlgoritmMutation(Generation)
Generation = des.GeneticAlgoritmImmigration(Generation)



# Figure of design
col=cb.qualitative.Set1_8.mpl_colors
plt.figure(figsize=(7,5))
for x in range(order['Z'].shape[1]):
    plt.plot(np.arange(0,TR*order['Z'].shape[0],TR),order['Z'][:,x],color=col[x],lw=3,label="stimulus: "+str(x+1))
plt.legend(loc="upper right",frameon=False)
plt.xlim([0,L*ISI])
plt.ylim([-0.5,3.5])
plt.xlabel("seconds")
plt.ylabel("normalised expected BOLD")
plt.show()


col=cb.qualitative.Set1_8.mpl_colors
plt.figure(figsize=(7,5))
for x in range(order['Z'].shape[1]):
    pt = np.arange(0,TR*order['Z'].shape[0],TR)[order['X'][:,x]==1]
    plt.plot(pt,[x]*len(pt),'ro',color=col[x],label="stimulus: "+str(x+1))
plt.legend(loc="upper right",frameon=False)
plt.ylim([-0.5,3.5])
plt.xlabel("seconds")
plt.ylabel("normalised expected BOLD")
plt.show()
