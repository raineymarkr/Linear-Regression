import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from tabulate import tabulate

X = np.array([[3.2, 3.4, 3.3, 3.5], [3.4, 3, 3.7, 3.3], [2.8, 2.6, 3, 2.7]]).T


def generateSSValues(A):
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
    x1dot = np.mean(A[:,0])
    x2dot = np.mean(A[:,1])
    x3dot = np.mean(A[:,2])
    print(x1dot, x2dot, x3dot)
    m = A.shape[1]
    n = A.shape[0]
    
    factor1 = n * m - m
    
    print(f'n * m - m = {factor1}')
    print('C:')
    C = input()
    W = np.sqrt(1 / n) * float(C) * np.sqrt(SSW / (n * m - m))
    
    diffonetwopos = x1dot - x2dot + W
    diffonetwoneg = x1dot - x2dot - W
    
    diffonethreepos = x1dot - x3dot + W
    diffonethreeneg = x1dot - x3dot - W
    
    difftwothreepos = x2dot - x3dot + W
    difftwothreeneg = x2dot - x3dot - W
    
    table = [[f"μ1 - μ2: {diffonetwoneg:.3f} < μ <  {diffonetwopos:.3f}"], [f"μ1 - μ2: {diffonethreeneg:.3f} < μ <  {diffonethreepos:.3f}"],[f"μ1 - μ2: {difftwothreeneg:.3f}  < μ <  {difftwothreepos:.3f}"]]
    headers = ["Statistic", "Value"]
    print(tabulate(table, headers, tablefmt="grid"))

def generateTable(SSW, WithinSample, SSB, BetweenSample):
    table = [["Within Sample Sum of Squares:", WithinSample], ["Between Sample Sum of Squares:", BetweenSample],['F-stat:', BetweenSample/WithinSample]]
    headers = ["Statistic", "Value"]
    print(tabulate(table, headers, tablefmt="grid"))
    print()

if __name__ == "__main__":
 
    np.set_printoptions(precision=2)
    #menu(x,y, firstMenu)
    

    X = np.array(X)
    
    SSW, WithinSample, SSB, BetweenSample = generateSSValues(X)
    
    generateTable(SSW, WithinSample, SSB, BetweenSample)
    comparisons(X, SSW)