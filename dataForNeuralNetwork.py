# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:35:32 2020

@author: SKY_SHY
"""

def getfun1SINUS(x):
    return np.sin(x)

def getfun2POW5(x, ak=3, ak2 = -1, ak3 = 1, ak4 = 2):
    return ak*x**5 + ak2*x**3 + ak3*x + ak4

def getfun3POW6(x, ak=-1, ak2 = 1, ak3 = -1, ak4 = 1):
    return ak*x**6 + ak2*x**2 + ak3*x + ak4


#******************************************************************
    
def getfun4EXP(x, a):
    return np.e(x*a)

def getfun5LN(x, a):
    return np.logaddexp2(x*a)


import nn as nn

X = list(range(20))

Y = [getfun3POW6(x) for x in X]

NeuralNetwork = nn.Main(2, X,[Y], [3,1])

NeuralNetwork.main(0)


Ya = NeuralNetwork.Outputs[-1] 

Ya = [y for item in Ya for y in item]

Ysecond = [getfun2POW5(x) for x in X]

print("вектор ошибки", np.array(Ya) - np.array(Y))

plotting = nn.Plotting([Y, Ya], X, Ysecond, ["sin(ax)", "3x^5 - x^3 + x + 2"])
plotting.error()
plotting.plot()

