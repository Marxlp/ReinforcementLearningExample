# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 23:23:48 2017

@author: Notroot
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

class kArmedBandit():
    def __init__(self,k=2,epsilon=0.0,alpha = None,bias = 0.0,c = None):
        self.ArmNum = k
        self.bandit = np.random.randn(k)
        self.epsilon = epsilon
        self.bestArm = np.argmax(self.bandit)
        self.alpha = alpha
        self.bias = bias
        self.c = c
        
    def play(self,times):
        #store whether current arm choosen is the best arm, 0 for no, 1 for 1
        self.whetherBestArm = np.zeros(times + 1)
        self.ArmCounts = np.zeros(self.ArmNum)
        self.q_star = np.zeros(self.ArmNum) + self.bias
        self.rewardCum = np.zeros(times + 1)
        self.Q = np.zeros(times + 1)
        update = self.updateFunc()
        for i in range(1,times+1):
            update(i)        
        self.Q[1:] = self.rewardCum[1:]/np.arange(1,times+1)
        
    def showResult(self):
        print 'bandit\n',self.bandit
        print 'q_star',self.q_star
        
    
    def setEpsilon(self,epsilon):
        self.epsilon = epsilon
    
    def setAlpha(self,alpha):
        self.alpha = alpha
        
    def setBias(self,bias):
        self.bias = bias
    
    def setC(self,c):
        self.c = c
    
    def getQ(self):
        return self.Q
    
    def getWhetherBestArm(self):
        return self.whetherBestArm
    
    def updateFunc(self):
        '''return different methods'''
        def averageSample(i):
            rnd = np.random.random()
            if self.epsilon > rnd:
                currentArm = np.random.choice(np.arange(self.ArmNum))
            else:
                currentArm = np.argmax(self.q_star)
            if currentArm == self.bestArm:
                self.whetherBestArm[i] = 1
            current_Reward = np.random.randn() + self.bandit[currentArm]
            self.ArmCounts[currentArm] += 1
            self.q_star[currentArm] += 1.0/self.ArmCounts[currentArm] *\
                                (current_Reward  - self.q_star[currentArm]) 
            self.rewardCum[i] = self.rewardCum[i-1] + current_Reward

        def stepSize(i):
            rnd = np.random.random(); 
            if self.epsilon > rnd:
                currentArm = np.random.choice(np.arange(self.ArmNum))
            else:
                currentArm = np.argmax(self.q_star)
            if currentArm == self.bestArm:
                self.whetherBestArm[i] = 1
            current_Reward = np.random.randn() + self.bandit[currentArm]
            self.ArmCounts[currentArm] += 1
            self.q_star[currentArm] += self.alpha *\
                                (current_Reward  - self.q_star[currentArm]) 
            self.rewardCum[i] = self.rewardCum[i-1] + current_Reward
        
        def UBC(i):
            currentArm = np.argmax(self.q_star + self.c * np.sqrt(np.log(i)/self.ArmCounts)) 
            if currentArm == self.bestArm:
                self.whetherBestArm[i] = 1
            current_Reward = np.random.randn() + self.bandit[currentArm]
            self.ArmCounts[currentArm] += 1
            self.q_star[currentArm] += self.alpha *\
                                (current_Reward  - self.q_star[currentArm]) 
            self.rewardCum[i] = self.rewardCum[i-1] + current_Reward
            
        #check whether alpha and c were difined
        if self.alpha == None and self.c == None:
#            print 'averageSample'
            return averageSample
        elif self.alpha == None:
#            print 'UBC'
            return UBC
        else:
#            print 'stepSize'
            return stepSize
            
    def plotRewardDistribution(self):
        fig,ax = plt.subplots() 
        ax.set_title(r"$q_*$ distribution")
        ax.set_xlabel(u"Action")
        ax.set_ylabel(u"Award")
        data = [sorted(np.random.normal(miu,1,300)) for miu in self.bandit]
        ax.violinplot(data,showmeans=False,showmedians=False,showextrema=False)
        for i,miu in enumerate(self.bandit):
            ax.hlines(miu,i+0.7,i+1.3)
            ax.text(i+1.5,miu,r'$q_*$'+r'('+str(i+1)+r')')
        ax.set_xticks(np.arange(self.ArmNum+1))
        plt.show()
    
    def plotPlayResult(self):
        fig,ax = plt.subplots()
        ax.set_title(r"Average Award")
        ax.set_xlabel(u"step")
        ax.set_ylabel(u"Award")
        Q1 = self.getQ()
        T = np.arange(0,len(Q1))
        ax.plot(T,Q1,'g-')
        plt.show()
        
def test1():
    
    TenArmed = kArmedBandit(10,0.01)
    TenArmed.plotRewardDistribution()
    fig,ax = plt.subplots()
    TenArmed.play(10000)
    TenArmed.showResult()
    Q1 = TenArmed.getQ()
    
    TenArmed.setEpsilon(0.1)
    TenArmed.play(10000)
    TenArmed.showResult()
    Q2 = TenArmed.getQ()
    
    TenArmed.setEpsilon(0.0)
    TenArmed.play(10000)
    TenArmed.showResult()
    Q3 = TenArmed.getQ()
    
    T = np.arange(0,len(Q1))
    ax.plot(T,Q1,'g-',label = r'$\epsilon$=0.01')
    ax.plot(T,Q2,'r-',label =r'$\epsilon$=0.1' )
    ax.plot(T,Q3,'b-',label =r'$\epsilon$=0' )
    ax.legend(loc='right')
    ax.set_xlabel('step')
    ax.set_ylabel('Average Award')

def test2():
    fig1,ax1 = plt.subplots()
    repeatTimes,stepTimes = 1000,3000
    #The average reward of all repeat tests with epsilon 0.0
    Q1 = np.zeros(1+stepTimes)
    #The time get the best armd
    count1 = np.zeros(1+stepTimes)
    
    Q2 = np.zeros(1+stepTimes)
    count2 = np.zeros(1+stepTimes)

    Q3 = np.zeros(1+stepTimes)
    count3 = np.zeros(1+stepTimes)
    
    for i in range(1,repeatTimes+1):        
        TenArmed = kArmedBandit(10,0.0)
        TenArmed.play(stepTimes)
        
        if i%200 == 0:
            print 'repeat times,epsilon:',i,TenArmed.epsilon
            TenArmed.showResult()
            
        Q1 = (Q1 * (i - 1) + TenArmed.getQ())/i
        count1 += TenArmed.getWhetherBestArm()
        
        #Set new epsilon
        TenArmed.setEpsilon(0.01)
        TenArmed.play(stepTimes)
        Q2 = (Q2 * (i - 1) + TenArmed.getQ())/i
        count2 += TenArmed.getWhetherBestArm()
        if i%200 == 0:
            print 'epsilon:',TenArmed.epsilon
            TenArmed.showResult()
        #set new epsilon
        TenArmed.setEpsilon(0.1)
        TenArmed.play(stepTimes)
        Q3 = (Q3 * (i - 1) + TenArmed.getQ())/i
        count3 += TenArmed.getWhetherBestArm()
        if i%200 == 0:
            print 'epsilon:',TenArmed.epsilon
            TenArmed.showResult()
            
    T = np.arange(0,len(Q1))
    ax1.plot(T,Q1,'b-',label =r'$\epsilon$=0' )
    ax1.plot(T,Q2,'g-',label = r'$\epsilon$=0.01')
    ax1.plot(T,Q3,'r-',label = r'$\epsilon$=0.1')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average reward')
    ax1.legend(loc = 'right')    
    
    fig2,ax2 = plt.subplots()
    count1[1:] = count1[1:]/stepTimes
    count2[1:] = count2[1:]/stepTimes
    count3[1:] = count3[1:]/stepTimes
    ax2.plot(T,count1*100,'b-',label = r'$\epsilon$=0')
    ax2.plot(T,count2*100,'g-',label = r'$\epsilon$=0.01')
    ax2.plot(T,count3*100,'r-',label = r'$\epsilon$=0.1')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Optimal action')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
    ax2.legend(loc = 'right')
    plt.show()
    
def test3():
    fig1,ax1 = plt.subplots()
    fig2,ax2 = plt.subplots()
    repeatTimes,stepTimes = 1000,3000
    #The average reward of all repeat tests with epsilon 0.0
    Q1 = np.zeros(1+stepTimes)
    #The time get the best armd
    count1 = np.zeros(1+stepTimes)
    
    Q2 = np.zeros(1+stepTimes)
    count2 = np.zeros(1+stepTimes)
    
    for i in range(1,repeatTimes+1):        
        TenArmed = kArmedBandit(10,0.1)
        TenArmed.setAlpha(0.1)
        TenArmed.play(stepTimes)
        
        if i%200 == 0:
            print 'repeat times,epsilon:',i,TenArmed.epsilon
            TenArmed.showResult()
              
        Q1 = (Q1 * (i - 1) + TenArmed.getQ())/i
        count1 += TenArmed.getWhetherBestArm()
        
        #Set new alpha
        TenArmed.setBias(5)
        TenArmed.setEpsilon(0.0)
        TenArmed.play(stepTimes)
        Q2 = (Q2 * (i - 1) + TenArmed.getQ())/i
        count2 += TenArmed.getWhetherBestArm()  
        
    T = np.arange(0,len(Q1)) 
    ax1.plot(T,Q1,'r-',T,Q2,'g--')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average reward')
    ax1.legend(loc = 'right')
    
    count1[1:] = count1[1:]/stepTimes
    count2[1:] = count2[1:]/stepTimes
    
    ax2.plot(T,count1*100,'r-',T,count2*100,'g--')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Optimal action')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
    ax2.legend(loc='right')
    plt.show()


if __name__ == '__main__':
    test2()
    