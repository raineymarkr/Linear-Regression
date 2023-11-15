import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from tabulate import tabulate

def getValues(A,b):
    print("n: ")
    n = int(input())
    print('k:')
    k = int(input())
    print('Input Matrix Values.')
    for i in range(n):
        temp = []
        for x in range(k+1):
            if x == 0:
                temp.append(1)
            else:
                new_x = float(input())
                temp.append(new_x)
        A.append(temp)
        print('Next Row...')
    print('Input Y Values.')
    for i in range(n):
        temp_y = float(input())
        b.append(temp_y)
    A = np.array(A)
    b = np.array(b)
    return A, b

def generateSSValues(A, b, pFlag):
    # SS Code
    sum = 0
    runsum = 0
    avg = []
    Xdotdot = 0
    rows = A.shape[0]
    cols = A.shape[1]
    for n in range(cols):
        for m in range(rows):
            sum += A[m,n]
            runsum += A[m,n]
        avg.append(sum/rows)
        sum = 0
    Xdotdot = runsum/ (rows*cols)

    SSW = 0
    diff2 = 0
    for n in range(cols):
        for m in range(rows):
            diff = A[m,n] - avg[n]
            SSW += (diff ** 2)
        factor = ((avg[n]-Xdotdot)**2)
        diff2 += factor
    SSB = rows * diff2 
    print(SSB)
    BetweenSample = SSB / (cols - 1)
                
    den = (rows * cols) - cols
    WithinSample = SSW/den
        
    return SSW, WithinSample, SSB, BetweenSample

def generatePlot(B,x,y,Title):
    yReg = B @ x.T
    #plt.scatter(x,y)
    plt.plot(x, yReg, color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(Title)
    plt.show()

def generateTable(SSW, WithinSample, SSB, BetweenSample):
    table = [["Within Sample Sum of Squares:", WithinSample], ["Between Sample Sum of Squares:", BetweenSample],['F-stat:', BetweenSample/WithinSample]]
    headers = ["Statistic", "Value"]
    print(tabulate(table, headers, tablefmt="grid"))
    print()

if __name__ == "__main__":
    X = [[220,251,226,246,260],[244,235,232,242,225],[252,272,250,238,256]]
    y = []
    
    #X,y = getValues(X,y)
    np.set_printoptions(precision=2)
    #menu(x,y, firstMenu)
    

    X = np.array(X).T
    
    SSW, WithinSample, SSB, BetweenSample = generateSSValues(X,y,True)
    generateTable(SSW, WithinSample, SSB, BetweenSample)
