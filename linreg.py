import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t
import os

x = [60,62,64,65,66,67,68,70,72,74]
y = [63.6,65.2,66,65.5,66.9,67.1,67.4,68.3,70.1,70]
#x=[]
#y=[]
def getValues(x,y):
    if os.path.exists(r'.\novfitness.csv'):
        df = pd.read_csv(r'.\novfitness.csv')
        
        dead_df = df[df['Exercise'] == 'Deadlift']
        date_df = dead_df['Date']
        weight_df = dead_df['Weight']
        reps_df = dead_df['Reps']
        volume_df = weight_df * reps_df
        last_date = ''
        daily_vol = 0
        daily_df = []
        for i in range(len(dead_df)):
            if last_date != '':
                if dead_df.get(i) == last_date:
                    daily_vol += volume_df.get(i)
                else:
                    daily_df.append(daily_vol)
                    last_date = date_df.get(i)
                    daily_vol = volume_df.get(i)
            else:
                last_date = date_df.get(i)
                daily_vol = volume_df.get(i)

    running = False
    global Flag
    Flag = False
    if(len(x) == 0):
        running = True
        print('Input X Values. Enter blank value to end.')
        while(running):
            new_x = input()
            if new_x != '':
                x.append(float(new_x))
            else:
                running = False

        if(len(y) == 0):
            running = True
        print('Input Y Values. Enter blank value to end.')
        while(running):
            new_y = input()
            if new_y != '':
                y.append(float(new_y))
            else:
                running = False
    if len(x) == len(y):
        x = np.array(x)
        y = np.array(y)
        Flag = False
        return x,y
    else:
        Flag == True
        return x,y
    
    

x,y = getValues(x,y)

# Graph
def generateSSValues(x,y, pFlag):
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

    #R²
    Rsq = (1 - SSR/SYY)
    #Linear Regression

    B = SxY / Sxx

    A = meanY - B * meanX

    
    if pFlag == True:
        print('SxY: ', SxY)
        print('Sxx: ', Sxx)
        print('SYY: ', SYY)
        print('SSR: ', SSR)
        print('R²: ', Rsq)

        print('A: ', A)
        print('B: ', B)
    return SxY, Sxx, SYY, SSR, Rsq, A, B

def generatePlot(A,B,x,y,Title):
    yReg = A + B * x
    plt.scatter(x,y)
    plt.plot(x, yReg, color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(Title)
    plt.show()

global firstMenu
firstMenu = True
def menu(x,y, firstMenu):
    
    while True:
        n = len(x)
        t_score = t.ppf(0.975, df=(n-2))
        if firstMenu:
            SxY, Sxx, SYY, SSR, Rsq, A,  B = generateSSValues(x,y,True)
            firstMenu = False
        else:
            SxY, Sxx, SYY, SSR, Rsq, A,  B = generateSSValues(x,y,False)    
        print('1: Run Tests')
        print('2: Show Plot')
        print('3: Residuals')
        print('0: Reinput Values')
        choice = input('?')
        if (choice == '1'):
            print('Which Test?')
            print('1: Test β')
            print('2: Test α')
            print('3: α+βx Confidence Interval (95%)')
            print('4: Y(x)')
            print('0: Go back')

            choice2 = input()
            if choice2 == '1':
                print("Beta: ")
                Beta = float(input())
                BTestStat = np.sqrt(((n-2)*Sxx)/SSR)*(B- Beta)
                
                
                print(t_score)
                if BTestStat < t_score:
                    print(f'{BTestStat} < {t_score}' )
                    print(f'Rejected @ 95%')
                else:
                    print(f'{BTestStat} > {t_score}' )
                    print('Not enough data to reject @ 95%')
            elif choice2 == '2':
                print("Alpha: ")
                Alpha = float(input())
                SumXSq = 0
                for i in range(n):
                    temp = x[i] * x[i]
                    SumXSq += temp
                ATestStat = np.sqrt((n*(n-2)*Sxx)/(SSR*SumXSq))*(A - Alpha)
                print('T: ', ATestStat)
            elif choice2 == '3':   
                print('X₀: ')
                xnaught = float(input())
                xmean = np.mean(x)
                xfactor = xnaught - xmean
                factor1 = np.sqrt((1/n) + (xfactor*xfactor)/Sxx) * np.sqrt(SSR/(n-2))
                t_score = t.ppf(0.975, df=(n-2))
                result_base = A + B*xnaught
                difference = t_score * factor1
                lower = result_base - difference
                upper = result_base + difference
                print(f'α+βx ∈ {result_base} +- {difference}')
                print(f'α+βx ∈ ( {lower}, {upper})')
            elif choice2 == '4':
                print('X₀: ')
                xnaught = float(input())
                xmean = np.mean(x)
                xfactor = xnaught - xmean
                factor1 = np.sqrt((((n+1)/n) + (xfactor*xfactor)/Sxx) * SSR/(n-2))
                t_score = t.ppf(0.975, df=(n-2))
                print(t_score, factor1)
                result_base = A + B*xnaught
                difference = t_score * factor1
                lower = result_base - difference
                upper = result_base + difference
                print(f'Y(X₀) ∈ {result_base} +- {difference}')
                print(f'Y(X₀) ∈ ( {lower}, {upper})')
        elif choice == '2':
            SxY, Sxx, SYY, SSR, Rsq, A, B = generateSSValues(x,y, True)
            generatePlot(A,B,x,y,'Data')
        elif choice == '3':
            res_y = []
            res_x = []
            n = len(y)
            for i in range(len(y)):
                num = y[i] - (A + B * x[i])
                den = np.sqrt(SSR/(n-2))
                res_y.append(num/den)
                res_x.append(i)
            res_x = np.array(res_x)
            res_y = np.array(res_y)
            SxY, Sxx, SYY, SSR, Rsq, A,  B  = generateSSValues(res_x, res_y, False)
            generatePlot(A,B,res_x,res_y,'Residuals')
        elif choice == '0':
            x,y = getValues([],[])
            generateSSValues(x,y, True)
if __name__ == "__main__":
    menu(x,y, firstMenu)
    
    while Flag == True:
        print("The lengths of x and y are not equal. Please reinput values.")
        x,y = getValues(x,y)

