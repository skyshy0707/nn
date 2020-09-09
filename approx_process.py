# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:41:45 2019

@author: SKY_SHY
"""

import numpy as np

import random

from numpy import linalg as la

from abc import ABCMeta, abstractmethod, ABC   
    
class Func():
    __metaclass__ = ABCMeta
    def __init__(self, a_X):
        super(Func, self).__init__()
        self.x = a_X
        self.a1 = 1
        self.a2 = 1
        self.a3 = 1
        self.a4 = 1
        
    def set_a1(self, a):
        self.a1 = a
        
    def set_a2(self, a):
        self.a2 = a
        
    def set_a3(self, a):
        self.a3 = a
    
    def set_a4(self, a):
        self.a4 = a
        
        

    @abstractmethod
    def getfun(self):
        pass
    
    @abstractmethod
    def getfunNAME(self):
        pass
    
    
class Case1(Func):
    
    def __init__(self, a_X):
        super().__init__(a_X)

        
    
    def getfun(self):
        return self.a1*np.sin(self.x)
    def getfunNAME(self):
        return str(self.a1) + 'sin(x)'

class Case2(Func):
    
    def __init__(self, a_X):
        super().__init__(a_X)
    
    def getfun(self):
        return self.a1*self.x+ self.a2
    
    def getfunNAME(self):
        return str(self.a1)+'x  +  ' + str(self.a2)

class Case3(Func):
    
    def __init__(self, a_X):
        super().__init__(a_X)
    
    def getfun(self):
        return np.sin(self.a1*self.x)
    def getfunNAME(self):
        return  'sin(' + str(self.a1) + 'x)'
    
class Case4(Func):
    def __init__(self, a_X):
        super().__init__(a_X)
        self.a1 = 3
        self.a2 = -1
        self.a4 = 2
    
    def getfun(self):
        return self.a1*self.x**5 + self.a2*self.x**3 + self.a3*self.x + self.a4
    
    def getfunNAME(self):
        return str(self.a1)+'x^5  +  ' + str(self.a2)+'x^3  +  ' + str(self.a3)+'x  +  ' + str(self.a4)
    
class Case5(Func):
    def __init__(self, a_X):
        super().__init__(a_X)
        self.a1 = -1
        self.a3 = -3
    
    def getfun(self):
        return self.a1*self.x**6 + self.a2*self.x**2 + self.a3*self.x + self.a4
    
    def getfunNAME(self):
        return str(self.a1)+'x^6  +  ' + str(self.a2)+'x^2  +  ' + str(self.a3)+'x  +  ' + str(self.a4)
    
class Case6(Func):
    def __init__(self, a_X):
        super().__init__(a_X)
    
    def getfun(self):
        return np.exp(self.a1*self.x)
    
    def getfunNAME(self):
        return  'e^' + str(self.a1) + 'x'
    
class Case7(Func):
    def __init__(self, a_X):
        super().__init__(a_X)
    
    def getfun(self):
        return np.log2(self.a1*self.x)
    
    def getfunNAME(self):
        return  'ln(' + str(self.a1) + 'x)'
    

class Case8(Func):
    def __init__(self, a_X):
        super().__init__(a_X)
    
    def getfun(self):
        return np.cos(self.a1*self.x)
    
    def getfunNAME(self):
        return  'cos(x)'

    
class Data_for_Noising(ABC):
        
    def __init__(self):
        self.E = 1
        self.scale_E = 1
        self.Disperion=0.1
        self.Method_distr_error ="normal"
        self.scale_Noise = 3
        self.percent_Noise = 10

    @abstractmethod   
    def settattr(self):
        pass
    
class uniform_noise_with_aE(Data_for_Noising):
    
    def __init__(self):
        super().__init__()
        
    def settattr(self):
        self.E = float(input("ВВЕДИТЕ ВЕЛИЧИНУ ОШИБКИ: "))
        self.scale_E = float(input("ВВЕДИТЕ МАСШТАБ ОШИБКИ: "))
        self.Method_distr_error = input("ВВЕДИТЕ ЗАКОН РАСПРЕДЕЛЕНИЯ ОШИБКИ: normal/uniform ")
        if self.Method_distr_error == "dispersion":
            self.Disperion= float(input("ВВЕДИТЕ ВЕЛИЧИНУ ДИСПЕРСИИ: "))
        
class uniform_noise_with_aE_plus_point_burst(Data_for_Noising):
    
    def __init__(self):
        super().__init__()
        
    def settattr(self):
        self.E = float(input("ВВЕДИТЕ ВЕЛИЧИНУ ОШИБКИ: "))
        self.scale_E = float(input("ВВЕДИТЕ МАСШТАБ ОШИБКИ: "))
        self.Method_distr_error = input("ВВЕДИТЕ ЗАКОН РАСПРЕДЕЛЕНИЯ ОШИБКИ: normal/uniform ")
        self.scale_Noise = float(input("ВВЕДИТЕ МАСШТАБ ВЫБРОСА, scale_Noise --- E*scale_Noise: "))
        if self.Method_distr_error == "dispersion":
            self.Disperion= float(input("ВВЕДИТЕ ВЕЛИЧИНУ ДИСПЕРСИИ: "))
    
    
    
class uniform_noise_with_aE_plus_percent_burst(Data_for_Noising):    
    
    def __init__(self):
        super().__init__()
        
    def settattr(self):
        self.E = float(input("ВВЕДИТЕ ВЕЛИЧИНУ ОШИБКИ: "))
        self.scale_E = float(input("ВВЕДИТЕ МАСШТАБ ОШИБКИ: "))
        self.Method_distr_error = input("ВВЕДИТЕ ЗАКОН РАСПРЕДЕЛЕНИЯ ОШИБКИ: normal/uniform ")
        self.scale_Noise = float(input("ВВЕДИТЕ МАСШТАБ ВЫБРОСА, scale_Noise --- E*scale_Noise: "))
        self.percent_Noise = int(input("ВВЕДИТЕ ПРОЦЕНТ ИСПОГАНЕНОЙ ШУМОМ ВЫБОРКИ: "))
        if self.Method_distr_error == "dispersion":
            self.Disperion= float(input("ВВЕДИТЕ ВЕЛИЧИНУ ДИСПЕРСИИ: "))
            

class point_burst(Data_for_Noising):    
    
    def __init__(self):
        super().__init__()
        
    def settattr(self):
        self.scale_Noise = float(input("ВВЕДИТЕ МАСШТАБ ВЫБРОСА, scale_Noise --- E*scale_Noise: "))

class percent_burst(Data_for_Noising):    
    
    def __init__(self):
        super().__init__()
        
    def settattr(self):
        self.scale_Noise = float(input("ВВЕДИТЕ МАСШТАБ ВЫБРОСА, scale_Noise --- E*scale_Noise: "))
        self.percent_Noise = int(input("ВВЕДИТЕ ПРОЦЕНТ ИСПОГАНЕНОЙ ШУМОМ ВЫБОРКИ: "))


    
        
        
        
    
class Data():
    
    "uniform noise with a E"
    "uniform noise with a E + point burst"
    "uniform noise with a E + % burst"
    "point burst"
    "% burst"
    
    def __init__(self, n, method_noising="% burst", method_generate_Xs="uniform", Interval_OX = 20, n_X = 15):
        self.n_func = n
        self.method_noising = method_noising
        self.method_generate_Xs = method_generate_Xs
        self.Interval_OX = Interval_OX 
        self.n_X = n_X
        self.X = self.generateXs()
        self.Y = self.choosing_function()
        self.perc_noising = 50
    
    
#*********ОПРЕДЕЛЯЕМ ИНТЕРФЕЙС ДЛЯ СОЗДАНИЯ ЗНАЧЕНИЙ АРГУМЕННТОВ ФУНКЦИИ ДЛЯ ЭКСПЕРИМЕНТА************
        
    class Generate_Xs(ABC):

        
        @abstractmethod
        def Xs(self):
            pass
        
    class From_0_to_n(Generate_Xs):
        
        def Xs(self, n, Interval):
            l = 2*Interval
            d = l/n
            return np.array([-Interval + d*i for i in range(n)])
        
    class Random(Generate_Xs):

        def Xs(self, n, Interval):
            return np.array(sorted([random.uniform(-Interval, Interval) for i in range(n)]))
    
#*********ОПРЕДЕЛЯЕМ ИНТЕРФЕЙС ДЛЯ СОЗДАНИЯ ШУМА В ВЫБОРКЕ************
            
    
    
    class Vybros(ABC):
        
        
        @abstractmethod
        def noising(self,):
            pass
        
        def disperion(self, unoised, noised):
            return np.mean((unoised - noised)**2)
    
    
        def Mu(self, unoised, disp):
            return (float(disp)/len(unoised))**0.5
        
        
    class UniformNoise(Vybros):
        


        def noising(self, Y, E, Noise , Disperion=0.1,  Method="normal"):
            Mistakes=[]
            if Method=="normal":
                return Y - np.array([random.uniform(E, -E) for item in Y])*Noise
            elif Method=="uniform":
                return Y -  np.array([random.normalvariate(E, Disperion) for item in Y])*Noise
            return Mistakes

    class Percentage(Vybros):
        
     
        
        def vybros(self, Y, amount, Mu_VAL):
            Mean_of_y_rand = np.mean(Y)
            Diff_mean_y_rand = [abs(i - Mean_of_y_rand) for i in Y]
            ind = Diff_mean_y_rand.index(min(Diff_mean_y_rand))
            Y = np.delete(Y, ind)
            Y = np.insert(Y, ind, max(Y) + (Mu_VAL*amount))
            return Y
        
        
        def disp_Mu(self, Y_unoised, Y_noised):
            disp = self.disperion(Y_unoised, Y_noised)
            return self.Mu(Y_unoised, disp)
        
        def noising(self, Y, amount, Mu_VAL, percent):
            Amount_vybes = int(len(Y)*percent/100)
            for i in range(Amount_vybes):
                Y = self.vybros(Y, amount, np.mean(Y))
            return Y

        def noisingE_with_Noise(self, Y_unoised, Y_noised, amount, Mu_VAL, percent):
            Amount_vybes = int(len(Y_noised)*percent/100)
            for i in range(Amount_vybes):
                Mu = self.disp_Mu(Y_unoised, Y_noised)
                Y_noised = self.vybros(Y_noised, amount, Mu)
            return Y_noised
        
    
    def noised_Y(self):
        
        if self.method_noising == "uniform noise with a E":
            Y_noised = self.UniformNoise()
            prts = uniform_noise_with_aE()
            prts.settattr()
            self.perc_noising = prts.percent_Noise
            E, Noise, Disperion,  Method = prts.E, prts.scale_E, prts.Disperion, prts.Method_distr_error
            return Y_noised.noising(self.Y.getfun(), E, Noise, Disperion,  Method)
        
        elif self.method_noising == "uniform noise with a E + point burst":
            Y_noised = self.UniformNoise()
            V = self.Percentage()
            prts = uniform_noise_with_aE_plus_point_burst()
            prts.settattr()
            self.perc_noising = prts.percent_Noise
            E, Noise, Disperion, Method , amount= prts.E, prts.scale_E, prts.Disperion, prts.Method_distr_error, prts.scale_Noise
            Y = Y_noised.noising(self.Y.getfun(), E, Noise, Disperion,  Method)
            disp = Y_noised.disperion(self.Y.getfun(), Y)
            Mu = Y_noised.Mu(self.Y.getfun(), disp)
            return V.vybros(Y, amount, Mu)
        
        elif self.method_noising == "uniform noise with a E + % burst":
            Y_noised = self.UniformNoise()
            V = self.Percentage()
            prts = uniform_noise_with_aE_plus_percent_burst()
            self.perc_noising = prts.percent_Noise
            E, Noise, Disperion, Method , amount, percent = prts.E, prts.scale_E, prts.Disperion, prts.Method_distr_error, prts.scale_Noise, prts.percent_Noise
            Y = Y_noised.noising(self.Y.getfun(), E, Noise, Disperion,  Method)
            disp = Y_noised.disperion(self.Y.getfun(), Y)
            Mu = Y_noised.Mu(self.Y.getfun(), disp)
            return V.noisingE_with_Noise(self.Y.getfun(), Y, amount, Mu, percent)
        
        elif self.method_noising == "point burst":
            V = self.Percentage()
            prts = point_burst()
            prts.settattr()
            self.perc_noising = prts.percent_Noise
            amount = prts.scale_Noise
            return V.vybros(self.Y.getfun(), amount, np.mean(self.Y.getfun()))
        
        elif self.method_noising == "% burst":
            V = self.Percentage()
            prts = percent_burst()
            prts.settattr()
            self.perc_noising = prts.percent_Noise
            amount, percent = prts.scale_Noise, prts.percent_Noise
            return V.noising(self.Y.getfun(), amount, np.mean(self.Y.getfun()), percent)
        
            
            
    "35+++++++++++++++++++++++++hnghngfbngfbn"
    def choosing_function(self):
        if self.n_func == "1":
            Funct = Case1(self.generateXs())
        elif self.n_func == "2":
            Funct = Case2(self.generateXs())
        elif self.n_func == "3":
            Funct = Case3(self.generateXs())
        elif self.n_func == "4":
            Funct = Case4(self.generateXs())
        elif self.n_func == "5":
            Funct = Case5(self.generateXs())
        elif self.n_func == "6":
            term_arr  = [self.Interval_OX+1 for i in range(self.n_X)]
            Xs = self.generateXs() + term_arr
            Funct = Case6(Xs)
            self.X = Xs
        elif self.n_func == "7":
            term_arr  = [self.Interval_OX+1 for i in range(self.n_X)]
            Xs = self.generateXs() + term_arr
            Funct = Case7(Xs)
            self.X = Xs
        return Funct
    
    def generateXs(self):
        if self.method_generate_Xs == "uniform":
            X = self.From_0_to_n()
            return X.Xs(self.n_X, self.Interval_OX)
        
        elif self.method_generate_Xs == "random":
            X = self.Random()
            return X.Xs(self.n_X, self.Interval_OX)
        
        
 
'''f = Data("5", "point burst")
f = Data("5")

X = f.generateXs()
Y = f.noised_Y()
unoised = f.Y.getfun()'''


