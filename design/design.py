import numpy as np
from numpy.linalg import inv
from numpy import transpose as t
from scipy.special import gamma
from collections import Counter
import pandas as pd

class Design(object):
    '''
    A class for an experimental design

    Parameters
    ----------
        ITI: float
            inter-trial interval
        TR: float
            repetition time
        L: integer
            number of trials
        P: numpy.array with floats
            probability of each trialtype
        C: numpy array with floats
            contrast matrix (each row = one contrast)
        rho: float
            AR(1) correlation coefficient
        Aoptimality: boolean, default True
            default usage of A-optimality criterion
            setting parameter to False results in D-optimality
        saturation: boolean, default True
            non-linearity in fMRI signal (see Kao et al.)
        resolution: float, default 0.1
            maximum resolution of design matrix
            has an impact of preciseness of convolved design with HRF
        weights
        G
        q
        I
    '''

    def __init__(self,ITI,TR,L,P,C,rho,weights,Aoptimality=True,saturation=True,resolution=0.1,G=20,q=0.01,I=4):
        self.ITI = ITI
        self.TR = TR
        self.L = L
        self.P = P
        self.S = len(P)
        self.C = C
        self.rc = C.shape[0]
        self.rho = rho
        self.Aoptimality = Aoptimality
        self.saturation = saturation
        self.resolution = resolution
        self.weights = weights
        self.G = G
        self.q = q
        self.I = I

        self.CreateTsComp()
        self.CreateLmComp()
        self.ComputeMaximumEfficiency()

    def CreateTsComp(self):
        self.duration = self.L*self.ITI #total duration (s)
        self.tp = int(self.duration/self.TR) # number of scans
        return self

    def CreateLmComp(self):
        # drift
        self.S = self.drift(np.arange(0,self.duration,self.TR)) #[tp x 1]

        # temporal autocorrelation
        self.var = np.eye(self.tp)
        self.R = np.eye(self.tp)
        for k in range(self.tp):
            if (k+1)<self.tp:
                self.R[k,k+1]=self.rho
                self.R[k+1,k]=self.rho
        self.V = np.dot(inv(self.var**2),inv(self.R**2))

        # orthogonal projection of whitened drift
        VS = self.V*self.S
        self.Pvs = reduce(np.dot,[VS,np.linalg.pinv(np.dot(t(VS),VS)),t(VS)])
        return self

    def GeneticAlgoritm(self):
        Generation = self.GeneticAlgoritmInitiate()
        Generation = self.GeneticAlgoritmCrossover(Generation)

        return Generation

    def GeneticAlgoritmInitiate(self):
        Generation = {
            'order' : [],
            'F' : [],
            'ID' : []
        }
        Generation = self.GeneticAlgoritmRandom(Generation,self.G)
        return Generation

    def GeneticAlgoritmRandom(self,Generation, r):
        for k in range(r):
            parent = self.RandomOrder(seed=k)
            parent = self.CreateDesignMatrix(parent)
            parent = self.ComputeEfficiency(parent)
            Generation['order'].append(parent['order'])
            Generation['F'].append(parent['F'])
            Generation['ID'].append(k)
        return Generation

    def GeneticAlgoritmCrossover(self,Generation): ## REPLACE OR ADD?
        CouplingRnd = np.random.choice(Generation['ID'],size=len(Generation['ID']),replace=True)
        CouplingRnd = [[CouplingRnd[i],CouplingRnd[i+1]] for i in np.arange(0,len(Generation['ID']),2)]
        for couple in CouplingRnd:
            changepoint = np.random.choice(self.L,1)[0]
            #create baby 1
            baby1_a = Generation['order'][couple[0]][:changepoint]
            baby1_b = Generation['order'][couple[1]][changepoint:]
            baby1 = {'order':baby1_a+baby1_b}
            baby1 = self.CreateDesignMatrix(baby1)
            baby1 = self.ComputeEfficiency(baby1)
            Generation['order'].append(baby1['order'])
            Generation['F'].append(baby1['F'])
            Generation['ID'].append(np.max(Generation['ID'])+1)
            #create baby 2
            baby2_a = Generation['order'][couple[1]][:changepoint]
            baby2_b = Generation['order'][couple[0]][changepoint:]
            baby2 = {'order':baby2_a+baby2_b}
            baby2 = self.CreateDesignMatrix(baby2)
            baby2 = self.ComputeEfficiency(baby2)
            Generation['order'].append(baby2['order'])
            Generation['F'].append(baby2['F'])
            Generation['ID'].append(np.max(Generation['ID'])+1)
        return Generation

    def GeneticAlgoritmMutation(self,Generation): ## REPLACE OR ADD?
        for order in Generation['order'][:self.G]:
            mutated = np.random.choice(self.L,int(round(self.L*self.q)),replace=False)
            mutatedorder = [np.random.choice(self.rc,1)[0] if ind in mutated else value for ind,value in enumerate(order)]
            mutatedbaby = {'order':mutatedorder}
            mutatedbaby = self.CreateDesignMatrix(mutatedbaby)
            mutatedbaby = self.ComputeEfficiency(mutatedbaby)
            Generation['order'].append(mutatedbaby['order'])
            Generation['F'].append(mutatedbaby['F'])
            Generation['ID'].append(np.max(Generation['ID'])+1)
        return Generation

    def GeneticAlgoritmImmigration(self,Generation):
        Generation = self.GeneticAlgoritmRandom(Generation,self.I)
        return Generation




    def RandomOrder(self,seed=np.random.randint(0,10**10)):
        '''
        Generate a random order of stimuli

        Parameters
        ----------
            seed: integer
                seed used for random generation

        Returns
        -------
            Design: dictionary
                dictionary with key 'order' which represents the order of the stimuli

        '''

        # generate random order

        mult = np.random.multinomial(1,self.P,self.L)
        Design = {"order" : [x.tolist().index(1) for x in mult]}

        return Design

    def CreateDesignMatrix(self,Design):
        '''
        Expand from order of stimuli to a fMRI timeseries

        Parameters
        ----------
            Design: dictionary
                Design['order']: dictionary with key 'order' generated by RandomOrder() or manual

        Returns
        -------
            Design: dictionary
                Design['X']: numpy array representing design matrix
                Design['Z']: numpy array representing convolved design matrix
        '''

        # upsample from trials to deciseconds

        tpX = int(self.duration/self.resolution) #total number of timepoints (upsampled)

        # expand random order to timeseries

        X_X = np.zeros([tpX,self.rc]) #upsampled Xmatrix
        XindStim = np.arange(0,tpX,self.ITI/self.resolution).astype(int) # index of stimuluspoints in upsampled Xmatrix
        for stimulus in range(self.rc):
            # fill
            X_X[XindStim,stimulus] = [1 if x==stimulus else 0 for x in Design["order"]]

        # convolve design matrix

        h0 = self.canonical(np.arange(0,self.duration,self.resolution))
        Z_X = np.zeros([tpX,self.rc])
        for stimulus in range(self.rc):
            Zinterim = np.convolve(X_X[:,stimulus],h0)[range(tpX)]
            ZinterimScaled = Zinterim/np.max(h0)
            if self.saturation==True:
                ZinterimScaled = [2 if x>2 else x for x in ZinterimScaled]
            Z_X[:,stimulus] = ZinterimScaled

        # downsample from deciseconds to scans

        XindScan = np.arange(0,tpX,self.TR/self.resolution).astype(int) # stimulus points
        X = np.zeros([self.tp,self.rc])
        Z = np.zeros([self.tp,self.rc])
        for stimulus in range(self.rc):
            X[:,stimulus] = X_X[XindScan,stimulus]
            Z[:,stimulus] = Z_X[XindScan,stimulus]

        # downsample and save in dictionary

        Design["X"] = X
        Design["Z"] = Z

        return Design

    def ComputeEfficiency(self,Design):
        '''
        Compute efficiency as defined in Kao, 2009 from fMRI timeseries

        Parameters
        ----------
            Design: dictionary
                Design['order']: dictionary with key 'order' generated by RandomOrder() or manual
                Design['X']: numpy array representing design matrix
                Design['Z']: numpy array representing convolved design matrix

        Returns
        -------
            Design: dictionary
                Design['Fe']: estimation efficiency
                Design['Fd']: detection power
                Design['Ff']: efficiency against psychological confounds
                Design['Fc']: optimality of probability of trials
        '''
        Design = self.FeCalc(Design)
        Design = self.FdCalc(Design)
        Design = self.FcCalc(Design)
        Design = self.FfCalc(Design)

        Design['Fe']=Design['Fe']/self.FeMax
        Design['Fd']=Design['Fd']/self.FdMax
        Design['Ff']=1-Design['Ff']/self.FfMax
        Design['Fc']=1-Design['Fc']/self.FcMax

        Design['F'] = np.sum(self.weights * np.array([Design['Fc'],Design['Fd'],Design['Fe'],Design['Ff']]))

        return Design

    def ComputeMaximumEfficiency(self):
        nulorder = [np.argmin(self.P)]*self.L
        NulDesign = {"order":nulorder}
        NulDesign = self.CreateDesignMatrix(NulDesign)
        self.FfMax = self.FfCalc(NulDesign)['Ff']
        self.FcMax = self.FcCalc(NulDesign)['Fc']
        self.FeMax = 1
        self.FdMax = 1

        return self

    def FeCalc(self,Design):
        W = Design['X']
        Cidentity = np.eye(self.rc)
        M = reduce(np.dot,[Cidentity,t(W),t(self.V),(np.eye(self.tp)-self.Pvs),self.V,W,Cidentity])
        if self.Aoptimality == True:
            Design["Fe"] = (self.rc/np.matrix.trace(M))
        else:
            Design["Fe"] = np.linalg.det(M)**(-1/self.rc)

        return Design

    def FdCalc(self,Design):
        W = Design['Z']
        M = reduce(np.dot,[self.C,t(W),t(self.V),(np.eye(self.tp)-self.Pvs),self.V,W,t(self.C)])
        if self.Aoptimality == True:
            Design["Fd"] = self.rc/np.matrix.trace(M)
        else:
            Design["Fd"] = np.linalg.det(M)**(-1/self.rc)
        return Design

    def FcCalc(self,Design):
        Q1 = np.zeros([self.rc,self.rc])
        Q2 = np.zeros([self.rc,self.rc])
        Q3 = np.zeros([self.rc,self.rc])
        for n in range(self.L):
            if n>0:
                Q1[Design['order'][n],Design['order'][n-1]] += 1
            if n>1:
                Q2[Design['order'][n],Design['order'][n-2]] += 1
            if n>2:
                Q3[Design['order'][n],Design['order'][n-2]] += 1
        Q1exp = np.zeros([self.rc,self.rc])
        Q2exp = np.zeros([self.rc,self.rc])
        Q3exp = np.zeros([self.rc,self.rc])
        for si in range(self.rc):
            for sj in range(self.rc):
                Q1exp[si,sj] = self.P[si]*self.P[sj]*(n-1)
                Q2exp[si,sj] = self.P[si]*self.P[sj]*(n-2)
                Q3exp[si,sj] = self.P[si]*self.P[sj]*(n-3)
        Q1match = np.sum(abs(Q1-Q1exp))
        Q2match = np.sum(abs(Q2-Q2exp))
        Q3match = np.sum(abs(Q3-Q3exp))
        Design["Fc"] = Q1match + Q2match + Q3match
        return Design

    # Ff frequencies
    def FfCalc(self,Design):
        trialcount = Counter(Design['order'])
        Pobs = [x[1] for x in trialcount.items()]
        Design["Ff"] = np.sum(abs(Pobs-self.L*self.P))
        return Design

    @staticmethod
    def drift(s):
        # second order Legendre polynomial
        # arguments: s = seconds after start
        ts = 1/2*(3*s**2-1)
        return ts

    @staticmethod
    def canonical(s,a1=6,a2=16,b1=1,b2=1,c=1/6,amplitude=1):
        #Canonical HRF as defined here: http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3318970/
        # arguments: s seconds
        gamma1 = (s**(a1-1)*b1**a1*np.exp(-b1*s))/gamma(a1)
        gamma2 = (s**(a2-1)*b2**a2*np.exp(-b2*s))/gamma(a2)
        tsConvoluted = amplitude*(gamma1-c*gamma2)
        return tsConvoluted
