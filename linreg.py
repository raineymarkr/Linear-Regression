import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = []
y = []

def importFile(file):
    df = pd.read_csv(r'C:\Users\raine\Downloads\bottle.csv')
    depth_df = df[df['Depthm'] <= 10]

    temp_df = depth_df['T_degC']
    salinity_df = depth_df['Salnty']

    combined_df = depth_df[['T_degC', 'Salnty']].dropna()

    x = []
    y = []

    x.extend(combined_df['T_degC'].tolist())
    y.extend(combined_df['Salnty'].tolist())


def getValues(x,y):

    running = False

    if(len(x) == 0):
        running = True
    print('Input X Label:')
    xLabel = input()
    print('Input X Values. Enter blank value to end.')
    while(running):
        new_x = input()
        if new_x != '':
            x.append(float(new_x))
        else:
            running = False

    if(len(y) == 0):
        running = True
    print('Input Y Label:')
    yLabel = input()
    print('Input Y Values. Enter blank value to end.')
    while(running):
        new_y = input()
        if new_y != '':
            y.append(float(new_y))
        else:
            running = False

    x = np.array(x)
    y = np.array(y)

    return x,y, xLabel, yLabel

x,y,xLabel,yLabel = getValues(x,y)

# Graph
def generateValues(x,y):
    meanX = np.mean(x)
    meanY = np.mean(y)

    #SxY
    SxYf1 = 0
    for i in range(len(x)):
        SxYf1 += x[i]*y[i]

    SxYf2 = len(x) * meanX * meanY

    SxY = SxYf1 - SxYf2

    #Sxx
    Sxxf1 = 0
    for i in range(len(x)):
        Sxxf1 += (x[i] * x[i])

    Sxxf2 = len(x) * meanX * meanX

    Sxx = Sxxf1 - Sxxf2

    #SYY
    SYYf1 = 0
    for i in range(len(x)):
        SYYf1 += (y[i] * y[i])

    SYYf2 = len(x) * meanY * meanY

    SYY = SYYf1 - SYYf2

    #SSR
    SSRf1 = Sxx * SYY
    SSRf2 = SxY*SxY

    SSR = (SSRf1 - SSRf2) / Sxx

    #Linear Regression

    B = SxY / Sxx

    A = meanY - B * meanX

    

    print('SxY: ', SxY)
    print('Sxx: ', Sxx)
    print('SYY: ', SYY)
    print('SSR: ', SSR)

    print('A: ', A)
    print('B: ', B)
    return SxY, Sxx, SYY, SSR, A, B

def generatePlot(A,B,x,y,xLabel,yLabel):
    yReg = A + B * x
    plt.scatter(x,y)
    plt.plot(x, yReg, color='red')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()


def menu(x,y):
    while True:    
        print('1: Run Tests')
        print('2: Show Plot')
        print('0: Reinput Values')
        choice = input('?')
        if (choice == '1'):
            print('Which Test?')
            print('1: Test β')
            print('2: Test α')
            print('3: α+βx')
            print('4: Y(x)')
            print('0: Go back')

            choice2 = input()
            if choice2 == '1':
                SxY, Sxx, SYY, SSR, A,  B = generateValues(x,y)
                print("Beta: ")
                Beta = float(input())
                BTestStat = np.sqrt(((len(y)-2)*Sxx)/SSR)*(B- Beta)
                print('T: ', BTestStat)
            elif choice2 == '2':
                SxY, Sxx, SYY, SSR, A, B = generateValues(x,y)
                print("Alpha: ")
                Alpha = float(input())
                SumXSq = 0
                for i in range(len(x)):
                    temp = x[i] * x[i]
                    SumXSq += temp
                ATestStat = np.sqrt((len(x)*(len(x)-2)*Sxx)/(SSR*SumXSq))*(A - Alpha)
                print('T: ', ATestStat)
            elif choice2 == '3':
                print('')
            elif choice2 == '4':
                print('')
        elif choice == '2':
            SxY, Sxx, SYY, SSR, A, B = generateValues(x,y)
            generatePlot(A,B,x,y,xLabel,yLabel)
        elif choice == '0':
            x,y,xLabel,yLabel = getValues([],[])
            generateValues(x,y)
if __name__ == "__main__":
    menu(x,y)