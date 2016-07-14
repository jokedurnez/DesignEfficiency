#!/usr/bin/python2.7
#
# Author: Joke Durnez
#
# Description: Generate m sequences
#
# Loosely translated from MATLAB to python
# from http://fmriserver.ucsd.edu/ttliu
#
# Date: 2016-07-13
#
#===========================================================

from __future__ import division
import numpy as np
import math
import pickle
import os
import sys

class TrialOrder(object):
    '''
    A class for an order of experimental trials

    Parameters
    ----------
        stimtypeno: integer
                    number of different stimulus types
    '''

    def __init__(self,stimtypeno):
        self.stimtypeno = stimtypeno

    def GenMseq(self,mLen,powerVal=None,baseVal=None,shift=None,whichSeq=None,userTaps=None):
        '''
        A function to generate msequences

        Parameters
        ----------
            mLen: integer
                desired length of sequence
            powerVal: integer
                the power of the msequence
            baseVal: integer
                the base value of the msequence
                (equal to the number of stimuli)
            shift: integer
                shift of the m-sequence
            whichSeq: integer
                index of the sequence desired in the taps file
            userTaps: list
                if user wants to specify own polynomial taps
        '''
        # read in taps file and count
        tapsfile = "/Users/Joke/Documents/Onderzoek/ProjectsOngoing/DesignEfficiency/MyDesign/design/taps.p"
        self.taps = pickle.load(open(tapsfile))

        # initate baseVal
        if not baseVal:
            baseVal = self.stimtypeno

        # initiate powerVal
        if not powerVal:
            minpow = math.log(mLen+1,baseVal)
            pos = self.taps[baseVal].keys()
            if pos > minpow:
                powerVal = np.min(pos[pos>minpow])
            else:
                powerVal = np.max(pos)

        # generate msequence
        self.Mseq(baseVal,powerVal,shift,whichSeq,userTaps)

        # length of sequence
        baseLen = baseVal**powerVal-1
        if mLen:
            if not mLen%baseLen == 0:
                print("Warning: The length of the msequence (base**power-1) is not equal to the requested length. It could be that the final probability of occurence for each stimulus type are not exactly equal.")
            rep = np.ceil(mLen/baseLen)
            if rep > 1:
                pos = taps[baseVal].keys()
                if pos > powerVal:
                    print("Warning: Because the length of the m-sequence (%s) is shorter than the requested length, the m-sequence will be repeated. To avoid too much repeating, you could choose a higher power.  The maximum power available is %s"%(baseLen,pos))
                else:
                    print("Warning: Because the length of the m-sequence (%s) is shorter than the requested length, the m-sequence will be repeated. This is the longest m-sequence known, so we can't avoid repeating the stimuli"%baseLen)
            ms = np.tile(self.ms,rep)
            self.ms = ms[:mLen]
        return self

    def Mseq(self,baseVal,powerVal,shift=None,whichSeq=None,userTaps=None):

        # compute total length
        bitNum = baseVal**powerVal-1

        # initiate register and msequence
        register = np.ones(powerVal)
        ms = np.zeros(bitNum)

        # select possible taps
        tap = self.taps[baseVal][powerVal]

        # if sequence is not given or false : random
        if (not whichSeq) or (whichSeq > len(tap) or whichSeq < 1):
            if whichSeq:
                print("You've asked a non-existing tap ! Generating at random.")
            whichSeq = math.ceil(np.random.uniform(0,1,1)*len(tap))

        # generate weights
        weights = np.zeros(powerVal)
        if baseVal == 2:
            weights = [1 for x in tap[int(whichSeq)] if x == 1]
        elif baseVal > 2:
            weights = tap[int(whichSeq)]
        else:
            print("You want at least 2 different stimulus types right? Now you asked for %s"%baseVal)

        if userTaps:
            weights = userTaps

        # generate msequence
        for i in range(bitNum):
            if baseVal == 4 or baseVal == 8 or baseVal == 9:
                tmp = 0
                for ind in range(len(weights)):
                    tmp = self.qadd(tmp,self.qmult(int(weights[ind]),int(register[ind]),baseVal),baseVal)
                ms[i] = tmp
            else:
                ms[i] = (np.dot(weights,register)+baseVal) % baseVal
            reg_shft = [x for ind,x in enumerate(register) if ind in range(powerVal-1)]
            register=[ms[i]]+reg_shft

        #shift
        if shift:
            shift = shift%len(ms)
            ms = np.append(ms[shift:],ms[:shift])

        # write to class
        self.ms = ms
        return self

    @staticmethod
    def qadd(a,b,base):
        # qadd(a,b,base)
        #
        # addition in a Galois Field for mod (power of prime)
        # Reference: K. Godfrey, Perturbation Signals for System #dentificaton,
        # 1993
        #
        # send comments and questions to ttliu@ucsd.edu

        if (a >= base or b >= base):
            print('qadd(a,b), a and b must be < %s'%(base))

        if base == 4:
            amat = np.array([
                [0,1,2,3],
            	[1,0,3,2],
            	[2,3,0,1],
            	[3,2,1,0],
            ])
        elif base == 8:
            amat = np.array([
                [0,1,2,3,4,5,6,7],
                [1,0,3,2,5,4,7,6],
                [2,3,0,1,6,7,4,5],
                [3,2,1,0,7,6,5,4],
                [4,5,6,7,0,1,2,3],
                [5,4,7,6,1,0,3,2],
                [6,7,4,5,2,3,0,1],
                [7,6,5,4,3,2,1,0]
            ])
        elif base == 9:
            amat = np.array([
                [0,1,2,3,4,5,6,7,8],
                [1,2,0,4,5,3,7,8,6],
                [2,0,1,5,3,4,8,6,7],
                [3,4,5,6,7,8,0,1,2],
                [4,5,3,7,8,6,1,2,0],
                [5,3,4,8,6,7,2,0,1],
                [6,7,8,0,1,2,3,4,5],
                [7,8,6,1,2,0,4,5,3],
                [8,6,7,2,0,1,5,3,4]
            ])
        else:
            print('qadd base %s not supported yet'%base)

        y = amat[a,b]
        return y

    @staticmethod
    def qmult(a,b,base):
        # qmult(a,b,base)
        #
        # multiplication in a Galois Field when base is a power of a prime number
        # Reference: K. Godfrey, Perturbation Signals for System Identificaton,
        # 1993
        #
        # send comments and questions to ttliu@ucsd.edu
        # Translated from qmult (Liu)

        if (a >= base or b >= base):
            print('qadd(a,b), a and b must be < %s'%(base))

        if base == 4:
            amult = np.array([
                [0,0,0,0],
                [0,1,2,3],
                [0,2,3,1],
                [0,3,1,2]
            ])
        elif base == 8:
            amult = np.array([
                [0,0,0,0,0,0,0,0],
                [0,1,2,3,4,5,6,7],
                [0,2,4,6,5,7,1,3],
                [0,3,6,5,1,2,7,4],
                [0,4,5,1,7,3,2,6],
                [0,5,7,2,3,6,4,1],
                [0,6,1,7,2,4,3,5],
                [0,7,3,4,6,1,5,2]
            ])
        elif base == 9:
            amult = np.array([
                [0,0,0,0,0,0,0,0,0],
                [0,1,2,3,4,5,6,7,8],
                [0,2,1,6,8,7,3,5,4],
                [0,3,6,4,7,1,8,2,5],
                [0,4,8,7,2,3,5,6,1],
                [0,5,7,1,3,8,2,4,6],
                [0,6,3,8,5,2,4,1,7],
                [0,7,5,2,6,4,1,8,3],
                [0,8,4,5,1,6,7,3,2]
        ])
        else:
            print('qmult base %s not supported yet'%base)

        y = amult[a,b]
        return y
