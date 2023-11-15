import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from tabulate import tabulate

def getValues(A):
    print("n: ")
    n = int(input())
    print('m:')
    k = int(input())
    print('Input Matrix Values.')
    for i in range(n):
        temp = []
        for x in range(k):
            new_x = float(input())
            temp.append(new_x)
        A.append(temp)
        print('Next Row...')
    A = np.array(A)
    return A

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

def comparisons(A,SSW):
    x1dot = np.mean(A[:],0)
    x2dot = np.mean(A[:],1)
    x3dot = np.mean(A[:],2)
    
    m = A.shape[1]
    n = A.shape[0]
    
    factor1 = n * m - m
    
    print(f'n * m - m = {factor1}')
    print('C:')
    C = input()
    W = np.sqrt(1 / n) * C * np.sqrt(SSW / (n * m - m))
    
    diffonetwopos = x1dot - x2dot + W
    diffonetwoneg = x1dot - x2dot - W
    
    diffonethreepos = x1dot - x3dot + W
    diffonethreeneg = x1dot - x3dot - W
    
    difftwothreepos = x2dot - x3dot + W
    difftwothreeneg = x2dot - x3dot - W
    
    table = [["μ1 - μ2:", diffonetwoneg + "< μ < " + diffonetwopos], ["μ1 - μ2:", diffonethreeneg + "< μ < " + diffonethreepos],["μ1 - μ2:", difftwothreeneg + "< μ < " + difftwothreepos]]
    headers = ["Statistic", "Value"]
    print(tabulate(table, headers, tablefmt="grid"))

def generateTable(SSW, WithinSample, SSB, BetweenSample):
    table = [["Within Sample Sum of Squares:", WithinSample], ["Between Sample Sum of Squares:", BetweenSample],['F-stat:', BetweenSample/WithinSample]]
    headers = ["Statistic", "Value"]
    print(tabulate(table, headers, tablefmt="grid"))
    print()

if __name__ == "__main__":
    X = []
    y = []
    
    X = getValues(X)
    np.set_printoptions(precision=2)
    #menu(x,y, firstMenu)
    

    X = np.array(X).T
    
    SSW, WithinSample, SSB, BetweenSample = generateSSValues(X,y,True)
    generateTable(SSW, WithinSample, SSB, BetweenSample)
