
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, spmatrix, solvers
from numpy import shape, reshape

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

def dot(x,y):
    return x[0]*y[0]+x[1]*y[1]

def linKernel2(x, y):
    return dot(x,y)

def gausKernel2(x,y,sigma):
    return np.exp(-np.linalg.norm([x[0]-y[0], x[1]-y[1]])**2/(2.*(sigma**2)))

def constructOptProblem(data, C, sigma):
    #Optimization Problem Construction
    print("Optimization Problem Constructing...")
    numData=len(data)
    pi=[]
    identity=spmatrix(1.0, range(numData), range(numData))
    pi=np.zeros((numData, numData))
    for i in range(numData):
        for j in range(numData):
            pi[i, j]=gausKernel2(data[i].p, data[j].p, sigma)*data[i].label*data[j].label
    temp=[]
    for i in range(len(data)):
        temp.append([float(data[i].label)])
    Pmat=matrix(pi)
    print(Pmat[0:10, 0:10])
    qmat=matrix([-1.0]*numData)
    Amat=matrix(temp)
    bmat=matrix(0.0)
    Gmat=matrix([-identity,identity])
    hmat=matrix([[0.0]*numData + [C]*numData])
    print("Optimization Problem Constructed")
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
def getClassificationValueNonLinear(alpha, bias, training_data, test_data, sigma):
    score=0.0
    for i in range(len(training_data)):
        score+=alpha[i]*training_data[i].label*gausKernel2(training_data[i].p, test_data, sigma)
    score+=bias
#    print ("Classification of test data [" +str(test_data[0])+", "+str(test_data[1])+"] is "+str(score))
    return score
def printValues(alpha, bias, training_data, test_data, sigma, softvectoridx):
    testscore=getClassificationValueNonLinear(alpha, bias, training_data, test_data, sigma)
    f=open("result.txt", "w")
    for i in softvectoridx:
        output_data="alpha of softvector ("+str(training_data[i].p1)+", "+str(training_data[i].p2)+") = "+str(alpha[i])+'\n'
        f.write(output_data)
    f.write('\n')
    output_data="bias is: "+str(bias)
    f.write(output_data)
    f.write('\n\n')
    output_data="score of point ("+str(test_data[0])+", "+str(test_data[1])+")="+str(getClassificationValueNonLinear(alpha, bias, training_data, test_data, sigma))
    f.write(output_data)
    f.close()
    return

def main():
    #variables
    C=1.0
    sigma=0.15

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
    print("File Loaded")

    #optimization problem
    optMatrix=constructOptProblem(hwData, C, sigma)
    sol=solvers.qp(optMatrix["P"],optMatrix["q"],optMatrix["G"],optMatrix["h"],optMatrix["A"],optMatrix["b"])
    print (sol)
    softvectoridx=findSoftVector(sol['x'])
    softvectorx=[lambda x: hwData[i].p1 for i in softvectoridx]
    softvectory=[lambda x: hwData[i].p2 for i in softvectoridx]
    softvector=[softvectorx[:],softvectory[:]]
    print(softvector)
    print(softvectoridx)
    bias=getBias(softvectoridx, hwData, sol['x'], sol['primal slack'], sigma)

    print ("bias: "+str(bias))

 #  testpoints=[]
 #  for i in np.linspace(0., 1.1, 10):
 #      for j in np.linspace(0., 1.1, 10):
 #          value=getClassificationValueNonLinear(sol['x'], bias, hwData, [i, j], sigma)
 #          if abs(value)<0.001:
 #              print ("found decision point")
 #          testpoints.append(value)

    printValues(sol['x'], bias, hwData, [0.5, 0.5], sigma, softvectoridx)
    #plot variables
    x1lst=[]; y1lst=[];x2lst=[];y2lst=[];
    for item in hwData:
        if item.label==1:
            x1lst.append(item.p1)
            y1lst.append(item.p2)
        else:
            x2lst.append(item.p1)
            y2lst.append(item.p2)
    plt.plot(x1lst, y1lst, 'ro', label="label=1")
    plt.plot(x2lst, y2lst, 'bo', label="label=-1")
    #mark support vectors
    for i in softvectoridx:
        plt.scatter(hwData[i].p1, hwData[i].p2, s=100, c="g")
    X1, X2=np.meshgrid(np.linspace (0, 1.03, 50), np.linspace(0.38, 1.01, 50))
    X=[[x1, x2] for x1, x2 in zip (np.ravel(X1), np.ravel(X2))]
    z=[]
    for meshpoints in X:
        z.append(getClassificationValueNonLinear(sol['x'], bias, hwData, meshpoints, sigma))
    Z=np.asarray(z).reshape(X1.shape)

    plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')

    plt.plot()
    plt.title("NonlinearSVM Data")
    plt.xlabel('p1');plt.ylabel('p2')
    plt.grid()

    plt.legend(bbox_to_anchor=(0., 1.00), loc=2, ncol=1, mode=None, borderaxespad=0.)
    plt.show()

#    plt.axis([0, 1.1, 0, 1.1])

if __name__=="__main__":
    main()