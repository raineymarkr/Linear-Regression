import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from tabulate import tabulate

def getValues(A,b):
    if A.size != 0:
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
    At = A.T
    AtA = At @ A
    factor1 = np.linalg.inv(AtA)
    factor2 = At @ b
    B = factor1 @ factor2
    
    ss_f1 = b.T @ b
    ss_f2 = B.T @ A.T @ b
    SSR = ss_f1 - ss_f2
    
    n, m = A.shape
    k = m - 1
    stdev = np.sqrt(SSR / (n - k - 1))
    meany = np.mean(b)
    SYY = sum((b - meany) ** 2)
    Rsq = 1 - (SSR / SYY)
    
    return B, SSR, stdev, Rsq, factor1

def generatePlot(B,x,y,Title):
    yReg = B @ x.T
    #plt.scatter(x,y)
    plt.plot(x, yReg, color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(Title)
    plt.show()

def generateTable(B, SSR, stdev, Rsq, XtXinv):
    # Ensure B and XtXinv are in a list format that tabulate can handle
    B_list = B.flatten().tolist() if isinstance(B, np.ndarray) else B
    
    table = [["B", B_list], ["SSR", SSR], ["Standard Deviation", stdev],[ "R²", Rsq]]
    headers = ["Statistic", "Value"]
    print(tabulate(table, headers, tablefmt="grid"))
    print(XtXinv)

if __name__ == "__main__":
    X = []
    y = []

    M = [
        [1, 44, 1.3, 250, 0.63, 18.1],
        [1, 33, 2.2, 115, 0.59, 19.6],
        [1, 33, 2.2, 75, 0.56, 16.6],
        [1, 32, 2.6, 85, 0.55, 16.4],
        [1, 34, 2.0, 100, 0.54, 16.9],
        [1, 31, 1.8, 75, 0.59, 17.0],
        [1, 33, 2.2, 85, 0.56, 20.0],
        [1, 30, 3.6, 75, 0.46, 16.6],
        [1, 34, 1.6, 225, 0.63, 16.2],
        [1, 34, 1.5, 255, 0.60, 18.5],
        [1, 33, 2.2, 175, 0.63, 18.7],
        [1, 36, 1.7, 170, 0.58, 19.4],
        [1, 33, 2.2, 75, 0.55, 17.6],
        [1, 34, 1.3, 85, 0.57, 18.3],
        [1, 37, 2.6, 90, 0.62, 18.8]
    ]

    X = [row[:-1] for row in M]
    y = [row[-1] for row in M]
    np.set_printoptions(precision=2)
    #menu(x,y, firstMenu)
    

    X = np.array(X)
    y = np.array(y)
    
    B, SSR, stdev, Rsq, XtXinv = generateSSValues(X,y,True)
    print(B, f"{SSR:.2f}, {stdev:.2f}, {Rsq:.2f}")
    
    generateTable(B, SSR, stdev, Rsq, XtXinv)
