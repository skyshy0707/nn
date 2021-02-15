# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 21:41:41 2016

@author: SKY_SHY
"""

def perceptronsValues2(arr):
    return [((1-np.exp(-arr[i]))/(1+np.exp(-arr[i]))) for i in range(len(arr))]



def perceptronsValues3(arr):
    return [(2/np.pi)*np.arctan(arr[i]) for i in range(len(arr))]


import numpy as np


class Initial_Data():
    
    def __init__(self, first_layer_perc, arrX, arrY=None, f = "X"):
        self.layer_perc = first_layer_perc
        self.arrX = arrX
        if f == "f":
            self.Grad = np.array([[[0 for row in range(self.layer_perc)] for j in range(2)]])
        else:
            self.Grad = np.array([[[0 for row in range(self.layer_perc)] for j in range(len(self.arrX[0])+1)],])
        
        self.arrY = arrY
        self.W = []
        
    def Ws_f_first(self):
        ws  = [[np.random.uniform(-(1/4), (1/4)) for i in range(self.layer_perc)] for i in range(2)]
        return ws
    
    
    def Ws(self):
        ws  = [[np.random.uniform(-(1/4), (1/4)) for i in range(self.layer_perc)] for i in range(len(self.arrX[0])+1)]
        return ws
    

        
    def WsdependsFrom(self):
        MaxR = max(self.arrY)
        MinR = min(self.arrY)
        ws  = [[np.random.uniform(MinR, MaxR) for i in range(self.layer_perc)] for i in range(len(self.arrX[0])+1)]
        return ws

    
    
    
    def add_W(self, arrW):
        self.W.append(arrW)
    
    
    
    def summWX(self):
        if isinstance(self.arrX[0], float):
            X = [[self.arrX[i]] for i in range(len(self.arrX))]
        else:
            X = self.arrX
        return [np.matmul(np.insert(X[i], 0, 1), self.W[-1]) for i in range(len(self.arrX))]

    

    def perceptronsValues(self):
        summ = self.summWX()
        return [[(2/np.pi)*np.arctan(summ[i][j]) for j in range(len(summ[i]))] for i in range(len(self.arrX))]


    def dP(self):
        summ = self.summWX()
        return [[(2/np.pi)*(1/((summ[i][j]**2)+1)) for j in range(len(summ[i]))] for i in range(len(summ))]

    
    def dXdW(self):
        dP = self.dP()
        dXdW = [[[dP[row][k]*self.arrX[row] for k in range(len(dP[row]))]] for row in range(len(dP))]
        dXdW = [np.insert(dXdW[i], 0, dP[i], axis = 0) for i in range(len(dXdW))]
        return dXdW
    

    #для промежуточных слоев (--- всех кроме первого и последнего)
    def dXdW_P(self):
        dP = self.dP()
        dXdW = [[[dP[row][k]*self.arrX[row][x] for k in range(len(dP[row]))] for x in range(len(self.arrX[row]))] for row in range(len(dP))]
        dXdW = [np.insert(dXdW[i], 0, dP[i], axis = 0) for i in range(len(dXdW))]
        return dXdW
    

    
    
    def dXdW_R(self):
        dXdW = [[[self.arrX[row][k] for i in range(self.layer_perc)] for k in range(len(self.arrX[row]))] for row in range(len(self.arrX))]
        dXdW = [np.insert(dXdW[i], 0, [1], axis = 0) for i in range(len(dXdW))]
        return dXdW 

    #пОСКОЛЬКУ ФУНКЦИЯ АКТИВАЦИИ - ВЕЩЕСТВЕННОЕ ЧИСЛО, ТО ЕЕ ВИД F(Y) = Y И ПРИ ВЗЯТИИ ПРО-ОЙ - ЭТО 1 
    
    
    def dYdX(self):
        return [[[self.W[-1][j][i]*1 for j in range(len(self.W[-1]))] for i in range(self.layer_perc)] for k in range(len(self.arrX))]
    

    def dYdX_Not_Real(self):
        dP = self.dP()
        return [[[self.W[-1][j][i]*dP[k][i] for j in range(len(self.W[-1]))] for i in range(self.layer_perc)] for k in range(len(self.arrX))]
    
    
    def dy(self, y_obs):
        if self.layer_perc ==1:
            dy = [[2*(self.arrY[j] - y_obs[j][i]) for i in range(self.layer_perc)] for j in range(len(self.arrY))]
        else:
            dy = [[[2*(self.arrY[j][i] - y_obs[j][i])] for i in range(self.layer_perc)] for j in range(len(self.arrY))]
        
        return dy
        
    
    def Inters_sub_layer(self, a_dy, a_dYdx):
        return [[sum([a_dy[j][i]*a_dYdx[j][i][k] for i in range(self.layer_perc)]) 
                 for k in range(len(self.W[-1]))] for j in range(len(self.arrX))]
    
    #ПЕРЕМЕННАЯ ДАННОЙ ФУНКЦИИ -- ВЫХОД ФУНКЦИИ  Inters_sub_layer  
    
    
    def computeGrad(self, a_SummErrors, f = 1):
        if f == 1:
            The_dXdW = self.dXdW()
        else:
            The_dXdW = self.dXdW_P()
        a_SummErrors = [a_SummErrors[i][1:] for i in range(len(a_SummErrors))]
        return [[sum([a_SummErrors[k][j] * The_dXdW[k][i][j] for k in range(len(self.arrX))]) for j in range(self.layer_perc)] for i in range(len(self.W[-1]))]
    
    
    def computeGrad_R(self, a_SummErrors):
        The_dXdW = self.dXdW_R()
        return [[sum([a_SummErrors[k][j] * The_dXdW[k][i][j] for k in range(len(self.arrX))]) for j in range(self.layer_perc)] for i in range(len(self.W[-1]))]

    def add_Grad(self, aGrad):
        self.delete()
        self.Grad = np.append(self.Grad, [aGrad], axis = 0)
        
    def delete(self):
        if len(self.Grad) >=3:
            for i in range(len(self.Grad)-2):
                self.Grad = np.delete(self.Grad, 0, 0)
        if len(self.W) >= 3:
            for i in range(len(self.W)-2):
                self.W.pop(0)



class NN():
    "*(len(self.Obj.arrX)*0.75)"
    
    def __init__(self, init_obj):
        self.__Treshold = 0.7
        self.__K = 0.075
        self.__FI = 0.2
        self.__A = 0.2
        self.AmoutError = 0.25
        self.Obj = init_obj
        self.E = np.array([[[self.__A for row in range(self.Obj.layer_perc)] for i in range(len(self.Obj.W[-1]))]])
        self.listExpCurrPastDervs = np.array([[[(1-self.__Treshold)*self.E[-1][i][row] for row in range(self.Obj.layer_perc)] for i in range(len(self.Obj.W[-1]))] for m in range(2)]) 
    
    
    def scale(self, m):
        return 1/m
        
    #θ
    def setTreshold(self, threshold):
        self.__Treshold  = threshold
        
    def setIncrConstantK(self, k):
        self.__K = k
        
    def setDecremetProportionFi(self, f):
        self.__FI = f
        
    def setAmountError(self, amountErr):
        self.AmoutError = amountErr
    def setAlpha(self, a):
        self.__A = a
    
    def delete(self):
        if len(self.listExpCurrPastDervs) >=3:
            for i in range(len(self.listExpCurrPastDervs)-2):
                self.listExpCurrPastDervs = np.delete(self.listExpCurrPastDervs, 0,0)
        if len(self.E) >= 3:
            for i in range(len(self.E)-2):
                self.E = np.delete(self.E, 0,0)
                
        
        
    #(1-θ)*δ(t)
    def fstTermOfpart_drvError(self):
        return np.array(self.Obj.Grad[-1]*(1-self.__Treshold))
     
    
    def add_expCurrPastDrvs(self):
        item = self.fstTermOfpart_drvError() + self.scdTermOfpart_drvError()
        self.listExpCurrPastDervs = np.append(self.listExpCurrPastDervs, [item], axis = 0)

    #θ*δ(t-1)

    def scdTermOfpart_drvError(self):
        return np.array(self.listExpCurrPastDervs[-2]*self.__Treshold)
    
    
    
    def Delta_Bar_Delta(self):
        self.add_expCurrPastDrvs()
        return [[self.listExpCurrPastDervs[-2][row][i] * self.Obj.Grad[-1][row][i] for i in range(self.Obj.layer_perc)] for row in range(len(self.Obj.W[-1]))]
    
    def add_dE(self,m, Max, tau):
        self.E = np.append(self.E, [self.amoutUpdateCoefs(m, Max, tau)], axis = 0)
		
    def amoutUpdateCoefs(self,m, Max, tau):
        delta_B_delta = self.Delta_Bar_Delta()
        "k**x, fi**x  = k**3.2, fi**3"
        return [[self.__K*tau if delta_B_delta[row][i]>0 else -self.__FI*self.E[-1][row][i]*tau if delta_B_delta[row][i]<0 else 0 for i in range(self.Obj.layer_perc)] for row in range(len(self.Obj.W[-1]))]
    
    def updateCoefficients(self, m, Max, tau):
        self.add_dE(m, Max, tau)
        dE = np.array(self.E[-1])
        W_updating = self.Obj.W[-1] + (dE* self.Obj.Grad[-1])
        print("nw W", W_updating,)
        
        "inputs: , self.Obj.arrX"
        self.Obj.add_W(W_updating)
        self.delete()
                  
import random
import matplotlib.pyplot as plt

import approx_process as ap
import networkx as nx
import time
class Main():

	def __init__(self, points, testPonts, N_perc_in_each_layer):#points as [(x, y)], where y - noised
		self.points = points
		self.testPoints = testPonts
		self.L = []
		self.Delta_bar_Delta = []
		self.Outputs = []
		self.n_perc = N_perc_in_each_layer
		self.n_last_layer = len(N_perc_in_each_layer) - 1
		self.N_layers = len(N_perc_in_each_layer)
		self.Update = []
		self.e = 0
		self.ERROR = []
		self.Outputs = []
		self._X0 = 1
		self._Xn = 20
		self.X_fit = self.X_for_fit()
		self.n_func = "1"
		self.Ymax = max(max(testPonts))
		self.Marks = [[""] + ["X="] if l == 0 
                       else ["Y=" for j in range(self.n_perc[l-1])] if l == self.N_layers 
                       else ["L" + str(l+1) +"_BIAS" ] + ["L" + str(l+1) + "_" + str(j+1) + "=" for j in range(self.n_perc[l-1])]  for l in range(self.N_layers+1)]
		self.G = nx.Graph()
		self.point_X = int(len(self.points)/2)
		self.pos = {} 
		self.M_X = np.mean(self.points)
		self.M_Y = np.mean([point for Set in self.testPoints for point in Set])
		self.N = len(self.points)
		self.n_func_1 = "1"
		self.n_func_2 = "2"
		self.Recognizer = []

	def set_interval_Xfit(self, X0, Xn):
		self._X0 = X0
		self._Xn = Xn
	
	def set_No_funct(self, No):
		self.n_func = str(No)
	
	def X_for_fit(self):
		n_pots = int(((self._Xn - self._X0) // (2*np.pi))*300)
		d = (self._Xn - self._X0)/n_pots
		print("n pots", n_pots)
		return np.array([i*d + d for i in range(n_pots)])
	
	def X_for_fit2(self):
		return np.array(sorted([float(random.uniform(self._X0,self._Xn)) for i in range(300)]))
	
	def set_No_functs(self, No):
		self.n_func_1 = str(No[0])
		self.n_func_2 = str(No[1])
		
	def update_graf(self,):
		plt.pause(0.1)
		self.G.clear()
		plt.clf()
	
	def graf(self,):
		plt.subplot(122)
		edges, colors = zip(*nx.get_edge_attributes(self.G,'weight').items())
		nx.draw(self.G, self.pos, edgelist=edges, edge_color=colors, with_labels=True)
		labels = nx.get_edge_attributes(self.G, 'weight')
		nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=labels)
	
	def color(self, w):
		if abs(w) > 16777215:
			return "#FFFFFF"
		else:
			return "#" + hex(int(abs(w)/16777215) + 1)[2:]

	def add_edges(self, ind):
		pos_s = {}
		if ind == 0:
			val_ent_arr = [1] + [self.L[ind].arrX[self.point_X]]
		else:
			val_ent_arr = [1] + self.L[ind-1].perceptronsValues()[self.point_X]
		
		if ind <self.n_last_layer:
			val_exit_arr = [1] + self.L[ind].perceptronsValues()[self.point_X]
		else:
			val_exit_arr = self.L[ind].summWX()[self.point_X]
        
		o = 2
		for i in range(len(self.Marks[ind])):
			vert_name_A = self.Marks[ind][i] + str(round(val_ent_arr[i], 4))
			x = ind*5
			if ind == 0:
				y = self.pos_y(ind) + i*o
			else:
				y = self.pos_y(ind) + (i+1)*o
			self.pos[vert_name_A] = [x, y]
			y = self.pos_y(ind+1)
			'''o +=2'''
			if ind == self.n_last_layer:
				iterat = range(0, len(self.Marks[ind+1]))
			else:
				iterat =  range(1, len(self.Marks[ind+1]))
			for j in iterat:
				m = 2
				vert_name_B = self.Marks[ind+1][j] + str(round(val_exit_arr[j], 4))
				
				self.pos[vert_name_B] = [x+5, y+2 + m*j]
				w = round(self.L[ind].W[-1][i][j-1], 4)
				self.G.add_edge(vert_name_A, vert_name_B, weight=w)
				self.G[vert_name_A][vert_name_B]['color'] = 'g'
				
	def pos_y(self, ind):
		M = max([len(mark) for mark in self.Marks])
		D = (M-1)*2
		if len(self.Marks[ind]) <M:
			d = (len(self.Marks[ind]) - 1)*2
			return (D-d)/2
		else:
			return 0
			
	def make_pos(self,):
		M = max([len(mark) for mark in self.Marks])
		D = (M-1)*2
		x = 0
		for mark in self.Marks:
			if len(mark) <M:
				d = (len(mark) - 1)*2
				y = (D-d)/2
			else:
				y = 0
			for v in mark:
				self.pos[v] = [x, y]
				y *= 2
			x *= 5
    
	def add_layer(self, l, i, init_coefs = True):
		if init_coefs:
			self.L.append(l)
			self.add_edges(i)
			
	def firstCoeff_Calc(self, N_example):
		X = self.points
		Y = self.testPoints[N_example]
		self.Ys_fit(X, Y, True)
		self.graf()
		self.update_graf()
		
	
	def compute_Dy(layer_num, Dx=None, obs_Y=None):
		if Dx:
			return Dx
		return self.L[layer_num].dy(obs_Y)
	
	def compute_dydx(layer_num,):
		if layer_num == self.n_last_layer:
			return self.L[layer_num].dYdX()
		return self.L[layer_num].dYdX_Not_Real()
	
		
	def compute_Dx(layer_num, Dy, dydx):
		return self.L[layer_num].Inters_sub_layer(Dy, dydx)

	def addGrad(layer_num):
		if layer_num == self.n_last_layer:
			dE = self.L[i].computeGrad_R(Dy)
		elif layer_num == 0:
			dE = self.L[i].computeGrad(Dy, 1)
		else:
			dE = self.L[i].computeGrad(Dy, 0)
		self.L[i].add_Grad(dE)
	
	def compute_obsY(layer_num):
		if layer_num == self.n_last_layer:
			return self.L[layer_num].summWX()
	
	def fitting(self, N_example):
		Dx=None
		for i in reversed(range(self.N_layers)):
			obs_Y = self.compute_obsY(i)
			Dy = self.compute_Dy(i, obs_Y, Dx)
			dydx = self.compute_dydx(i)
			Dx = self.compute_Dx(i, Dy, dydx)
			self.addGrad(i)

	def make_Delta_bar_Delta_obj(self):
		for i in range(self.N_layers):
			self.Delta_bar_Delta.append(NN(self.L[i]))
    
	def pow_2(self,i):
		if i == self.N_layers-1:
			return 0
		else:
			return 1
	#elif i == self.N_layers-1:
	#return self.N * (2**self.pow_2(i)) * self.M_Y * (w**Pow)'''
	def eta_n(self,i):
		w = 1/4
		Pow = self.N_layers-1 - i
		if i ==0:
			print("AMOUNT UPDATING WIGHTS first layer:", self.N * (2**self.pow_2(i)) * (self.M_Y**2) * (w**Pow-1) * self.M_X)
			return self.N * (2**self.pow_2(i)) * (self.M_Y**2) * (w**Pow-1) * self.M_X
		elif i == self.N_layers-1 and self.M_Y < 2:
			return 1000
		elif i == self.N_layers-1:
			return self.N * (2**self.pow_2(i)) * self.M_Y * (w**Pow)
		else:
			print("AMOUNT UPDATING WIGHTS sun layer:", self.N * (2**self.pow_2(i)) * (self.M_Y**2) * (w**Pow)*1)
			return self.N * (2**self.pow_2(i)) * (self.M_Y**2) * (w**Pow)*1
			
	def update(self):#ИЗМ. В СООТВЕТСТВИИ С КОЛ-ВОМ ПРЕДЫДУЩИХ СЛОЕВ!!
		for i in range(self.N_layers):
			p =np.prod(self.n_perc[i:])
			self.Delta_bar_Delta[i].updateCoefficients(np.prod(self.n_perc[:i+1]), 1/(self.Ymax/10), 1/(self.eta_n(i)*p))
			self.add_edges(i)	
	
	def paste_layer(self, X, i, f, Y = None):
		l = Initial_Data(self.L[i].layer_perc, X, Y, f)
		ws = self.L[i].W[-1]
		l.add_W(ws)
		return l
	
	def paste_outs(self, l, last_l = True):
		if last_l:
			return l.summWX()
		return l.perceptronsValues()
		
	def Youts(self, last_l, fit, init_coefs):
		if last_l and init_coefs:
			return fit
		return None
	
	def Ys_fit(self, X_fit, fit = None, init_coefs = False):
		outs = X_fit
		last_l = False
		f = "f"
		for i in range(self.N_layers):
			if i == self.n_last_layer:
				last_l = True
			if i != 0:
				f = "X"
			fit = self.Youts(last_l, fit, init_coefs)
			l = self.paste_layer(outs, i, fit, f)
			outs = self.paste_outs(l, last_l)
			self.add_layer(l, i, init_coefs)
		return [X[i][0] for i in range(len(X))]
	
	def Ys(self):
		X = None
		for i in range(self.N_layers):
			if i == self.n_last_layer:
				self.L[i].arrX = X
				Y = self.L[i].summWX()
			else:
				X = self.L[i].perceptronsValues()
		return Y
	
	def signalError(self, a_Ys, N_example):
		a_Ys = [a_Ys[i][0] for i in range(len(a_Ys))]
		e = sum((np.array(a_Ys) - np.array(self.testPoints[N_example]))**2)
		if e > self.e:
			return False
		return True
	
	def set_e(self, a_e):
		self.e = (a_e**2)*len(self.points)
	
	def error(self, a_Ys,  N_example):
		a_Ys = [a_Ys[i][0] for i in range(len(a_Ys))]
		return sum((a_Ys - self.testPoints[N_example])**2)
	
	def std(self, a_Ys,  N_example):
		a_Ys = [a_Ys[i][0] for i in range(len(a_Ys))]
		return np.std((a_Ys - self.testPoints[N_example])**2)
	
	def mean_abs_error(self, a_Ys,  N_example):
		a_Ys = [a_Ys[i][0] for i in range(len(a_Ys))]
		return np.mean((a_Ys - self.testPoints[N_example])**2)
	
	def addError(self, a_e):
		self.ERROR.append(a_e)
	
	def signalStop(self,):
		if len(self.ERROR)<5:
			return False
		else:
			if self.ERROR[-5]>self.ERROR[-4]>self.ERROR[-3]>self.ERROR[-2]<self.ERROR[-1]:
				return True
		return False
	
	def etalon_F(self , N_F):
		if N_F == "1":
			Funct = ap.Case1(self.X_fit)
		elif N_F == "2":
			Funct = ap.Case2(self.X_fit)
		elif N_F == "3":
			Funct = ap.Case3(self.X_fit)
		elif N_F == "4":
			Funct = ap.Case4(self.X_fit)
		elif N_F == "5":
			Funct = ap.Case5(self.X_fit)
		elif N_F == "6":
			Funct = ap.Case6(self.X_fit)
		elif N_F == "7":
			Funct = ap.Case7(self.X_fit)
		elif N_F == "8":
			Funct = ap.Case8(self.X_fit)
		return Funct.getfun()
	
	def plot(self, X, double_arr_Y):
		ax = plt.subplot(121)
		ax.plot(X, double_arr_Y[0], marker='o',color = 'blue')
		ax.plot(X, double_arr_Y[1], marker='o',color = 'magenta')
		ax.plot(X, double_arr_Y[2], marker='+', color = 'cyan')
		ax.plot(double_arr_Y[3], double_arr_Y[4], marker='+',color = 'red')
		ax.plot(self.points, self.testPoints[0], marker='+', color = 'green')
	
	def Plt(self, arr1x, arr2Y):
		self.graf()
		self.plot(arr1x, arr2Y)
		plt.show()
		self.update_graf()
	
	def noise(self, aY , noise_int, multer):
		obs = ap.Data("5",)
		obs.Y = aY
		noise  = obs.UniformNoise()
		Y = noise.noising(obs.Y, noise_int, multer)
		return Y
    
	def recognizer(self, arr):
		if abs(np.mean(arr[2] - arr[0])) < abs(np.mean(arr[2] - arr[1])):
			return "blue"
		else:
			return "magenta"
	
	def add_rec(self, rec):
		self.Recognizer.append(rec)

	def main(self, N_example):
		self.firstCoeff_Calc(N_example)
		self.make_Delta_bar_Delta_obj()
		count = 0
		YYs_etalon1 = self.noise(self.etalon_F(self.n_func_1), 0.1, 1)
		YYs_etalon2 = self.noise(self.etalon_F(self.n_func_2), 0.1, 1)
		while True:
			count +=1
			self.fitting(N_example)
			my_Ys = self.Ys()
			error = self.mean_abs_error(my_Ys, N_example)
			self.addError(error)
			YYs = self.Ys_fit(self.X_fit)
			if self.signalError(my_Ys, N_example):
				break
			if count>=2000:
				print("РЕКОМЕНДОВАННОЕ ЗНАЧЕНИЕ ОШИБКИ ДЛЯ УСТАНОВКИ ПОРОГА ПОИСКА ГЛОБАЛЬНОГО МИНИМУМА:  ", min(self.ERROR))
				break
			self.update()
            
			Yc = [my_Ys[i][0] for i in range(len(my_Ys))]
			self.Plt(self.X_fit, [YYs_etalon1, YYs_etalon2, YYs, self.points, Yc])
			self.add_rec(self.recognizer([YYs_etalon1, YYs_etalon2, YYs]))
            
		print(self.error(my_Ys, N_example))
		self.Outputs.append(my_Ys)
		print("error:", self.ERROR, "num iteratons:", count)

class Plotting():#строит разл. графики при одинаковых значений x

	def __init__(self,List_orditates,  Xs, SecondYs, names):
		self.Y = List_orditates + [SecondYs]
		self.X = Xs
		self.funNames = names
	
	def plot(self):
		fig = plt.figure()
		
		for i in range(len(self.Y)):
			ax = fig.add_subplot(111)
			ax.plot(self.X, self.Y[i], marker='o',)
			plt.show()
	
	def error(self):
		d_Ya_Ysecond = abs(np.mean(np.array(self.Y[1]) - np.array(self.Y[2])))
		d_Ya_Y = abs(np.mean(np.array(self.Y[1]) - np.array(self.Y[0])))
		if d_Ya_Ysecond < d_Ya_Y:
			return self.funNames[1]
		elif d_Ya_Ysecond > d_Ya_Y:
			return self.funNames[0]