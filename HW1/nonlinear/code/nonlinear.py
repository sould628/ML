
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, spmatrix, solvers


class hdata(object):
    def __init__(self):
        pass
    def describe(self):
        pass
class h2DLdata(hdata):
    def __init__(self, p1, p2, label):
        self.p1=p1
        self.p2=p2
        self.label=label
        self.p=[p1,p2]
        self.dimension=2
    def describe(self):
        print ("p1: "+str(self.p1)+", p2: "+str(self.p2)+", label: "+str(self.label))

def linKernel2(x, y):
    return x[0]*y[0]+x[1]*y[1]

def gausKernel2(x,y,sigma):
    return np.exp(-((x[0]-y[0])**2+(x[1]-y[1])**2)/(2.*sigma**2))

def constructOptProblem(data, C, sigma):
    #Optimization Problem Construction
    numData=len(data)
    pi=[]
    identity=spmatrix(1.0, range(numData), range(numData))
    for i in range(len(data)):
        coli = []
        for j in range(len(data)):
            coli.append(gausKernel2(data[i].p, data[j].p, sigma)*data[i].label*data[j].label)
        pi.append(coli)
    temp=[]
    for i in range(len(data)):
        temp.append([float(data[i].label)])
    Pmat=matrix(pi)
    qmat=matrix([-1.0]*numData)
    Amat=matrix(temp)
    bmat=matrix(0.0)
    Gmat=matrix([-identity,identity])
    hmat=matrix([[0.0]*numData + [C]*numData])
    return {"P":Pmat, "q":qmat, "G":Gmat, "h":hmat, "A":Amat, "b":bmat}
def getWeightLin(data, alpha):
    numData=len(data);
    w=[0]*data[0].dimension
    for i in range(numData):
        alpha_mul_y=alpha[i]*data[i].label
        w[0]+=alpha_mul_y*data[i].p1
        w[1]+=alpha_mul_y*data[i].p2
    return w
def findSoftVector(x):
    result=[]
    for i in range(len(x)):
        if x[i]>0.00001 or x[i]<-0.00001:
            result.append(i)
    return result
def getBias(softvectoridx, data, alpha, slack, sigma):
    result=0.0
    i=softvectoridx[0]
    result=data[i].label*(1-slack)
    sum=0.0
    for j in range(len(data)):
        sum+=alpha[j]*data[j].label*gausKernel2(data[j].p, data[i].p, sigma)
    result-=sum
    return result



def main():
    #variables
    C=1.0
    sigma=15.0

    hwData=[]
    #ReadFile
    file_location="./../file/data-ex2.txt"
    f=open(file_location, "r")
    lines=f.readlines()
    for line in lines:
        word=line.split()
        label=int(word[0])
        p1=float(word[1][2:])
        p2=float(word[2][2:])
        hwData.append(h2DLdata(p1, p2, label))
    f.close()
    #ReadFile and Saved data in hwData

    #optimization problem
    optMatrix=constructOptProblem(hwData, C, sigma)
    sol=solvers.qp(optMatrix["P"],optMatrix["q"],optMatrix["G"],optMatrix["h"],optMatrix["A"],optMatrix["b"])
    print (sol)
    w=getWeightLin(hwData, sol['x'])
    softvectoridx=findSoftVector(sol['x'])
    print(softvectoridx)
    bias=getBias(softvectoridx, hwData, sol['x'], sol['primal slack'], sigma)


    print( "weight: "+str(w))
    print ("bias: "+str(bias))

    #plot variables
    x1lst=[]; y1lst=[];x2lst=[];y2lst=[];
    xline=np.linspace(0, 5, 100)
    for item in hwData:
        if item.label==1:
            x1lst.append(item.p1)
            y1lst.append(item.p2)
        else:
            x2lst.append(item.p1)
            y2lst.append(item.p2)
    plt.plot(x1lst, y1lst, 'ro', label="label=1")
    plt.plot(x2lst, y2lst, 'bo', label="label=-1")
    plt.plot(xline, -(xline*w[0]+bias)/w[1], 'k')
    plt.plot()
    plt.title("LinearSVM Data")
    plt.xlabel('p1');plt.ylabel('p2')
    plt.grid()
    plt.axis([0, 1.1, 0, 1.1])
    plt.legend(bbox_to_anchor=(0., 1.00), loc=2, ncol=1, mode=None, borderaxespad=0.)
    plt.show()



if __name__=="__main__":
    main()