from __future__ import division
import numpy as np
import palettable.colorbrewer as cb
import matplotlib.pyplot as plt

from design import design
from design import mseq


ISI = 2
TR = 2
L = 255
p = np.array([1/3,1/3,1/3])
C = np.array([
    [ 1, -1,  0],
    [ 1,  0, -1],
    [ 0,  1, -1],
])
rho = 0.3 #temporal autocorrelation, assume AR(1)
weights = np.array([1/4,1/4,1/4,1/4])
G=20


reload(design)
des = design.GeneticAlgorithm(ISI,TR,L,p,C,rho,weights)
des.GeneticAlgorithm()
des.opt['FScores']

reload(design)
des = design.GeneticAlgorithm(ISI,TR,L,p,C,rho,weights)
Generation = des.GeneticAlgorithmInitiate()

################
# Optimization #
################

col=cb.qualitative.Set1_8.mpl_colors
plt.figure(figsize=(7,5))
plt.plot(des.opt['FScores'],color=col[3],lw=2,label="")
plt.legend(loc="upper right",frameon=False)
#plt.xlim([0,L*ISI])
plt.ylim([0.8,1])
plt.xlabel("seconds")
plt.ylabel("normalised expected BOLD")
plt.show()





####################
# Inspect best one #
####################


# Figure of design
col=cb.qualitative.Set1_8.mpl_colors
plt.figure(figsize=(7,5))
for x in range(Generation['Z'].shape[1]):
    plt.plot(Generation['Z'][:,x],color=col[x],lw=2,label="stimulus: "+str(x+1))
plt.legend(loc="upper right",frameon=False)
#plt.xlim([0,L*ISI])
plt.ylim([-0.5,3.5])
plt.xlabel("seconds")
plt.ylabel("normalised expected BOLD")
plt.show()








###################

sequence = mseq.TrialOrder(4)
sequence.GenMseq(3,30)
sequence.ms

###################

# Figure of design
col=cb.qualitative.Set1_8.mpl_colors
plt.figure(figsize=(7,5))
for x in range(Generation['Z'].shape[1]):
    plt.plot(Generation['Z'][:,x],color=col[x],lw=3,label="stimulus: "+str(x+1))
plt.legend(loc="upper right",frameon=False)
#plt.xlim([0,L*ISI])
plt.ylim([-0.5,3.5])
plt.xlabel("seconds")
plt.ylabel("normalised expected BOLD")
plt.show()

### drift
col=cb.qualitative.Set1_8.mpl_colors
x = np.arange(0,100,2) #seconds of each scan
y = design.GeneticAlgorithm.drift(x)
plt.figure(figsize=(7,5))
plt.plot(x,y,color=col[0],lw=3,label="drift")
plt.legend(loc="upper right",frameon=False)
plt.xlim([0,100])
#plt.ylim([-0.5,3.5])
plt.xlabel("seconds")
plt.ylabel("drift")
plt.show()


col=cb.qualitative.Set1_8.mpl_colors
plt.figure(figsize=(7,5))
for x in range(Generation['Z'].shape[1]):
    pt = np.arange(0,TR*Generation['Z'].shape[0],TR)[Generation['X'][:,x]==1]
    plt.plot(pt,[x]*len(pt),'ro',color=col[x],label="stimulus: "+str(x+1))
plt.legend(loc="upper right",frameon=False)
plt.ylim([-0.5,3.5])
plt.xlabel("seconds")
plt.ylabel("normalised expected BOLD")
plt.show()




col=cb.qualitative.Set1_8.mpl_colors
plt.figure(figsize=(7,5))
plt.plot(Generation['FScores'],color=col[0],lw=3,label="optimisation")
plt.legend(loc="upper right",frameon=False)
plt.xlabel("cycles")
plt.ylabel("F")
plt.show()
