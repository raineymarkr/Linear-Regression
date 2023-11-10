import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t
import os

x = []
y = []

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
def generateValues(x,y, pFlag):
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

def generatePlot(A,B,x,y):
    yReg = A + B * x
    plt.scatter(x,y)
    plt.plot(x, yReg, color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
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
            print('3: α+βx Confidence Interval (95%)')
            print('4: Y(x)')
            print('0: Go back')

            choice2 = input()
            if choice2 == '1':
                SxY, Sxx, SYY, SSR, Rsq, A,  B = generateValues(x,y,False)
                print("Beta: ")
                Beta = float(input())
                BTestStat = np.sqrt(((len(y)-2)*Sxx)/SSR)*(B- Beta)
                
                t_score = t.ppf(0.95, df=(len(x)-2))
                if BTestStat < t_score:
                    print(f'{BTestStat} < {t_score}' )
                    print(f'Rejected @ 95%')
                else:
                    print(f'{BTestStat} > {t_score}' )
                    print('Not enough data to reject @ 95%')
            elif choice2 == '2':
                SxY, Sxx, SYY, SSR, Rsq, A, B = generateValues(x,y, False)
                print("Alpha: ")
                Alpha = float(input())
                SumXSq = 0
                for i in range(len(x)):
                    temp = x[i] * x[i]
                    SumXSq += temp
                ATestStat = np.sqrt((len(x)*(len(x)-2)*Sxx)/(SSR*SumXSq))*(A - Alpha)
                print('T: ', ATestStat)
            elif choice2 == '3':   
                print('X₀: ')
                SxY, Sxx, SYY, SSR, Rsq, A, B = generateValues(x,y, False)
                xnaught = float(input())
                xmean = np.mean(x)
                xfactor = xnaught - xmean
                factor1 = np.sqrt((1/len(x)) + (xfactor*xfactor)/Sxx) * np.sqrt(SSR/(len(x)-2))
                t_score = t.ppf(0.95, df=(len(x)-2))
                result_base = A + B*xnaught
                difference = t_score * factor1
                lower = result_base - difference
                upper = result_base + difference
                print(f'α+βx ∈ ( {lower}, {upper})')
            elif choice2 == '4':
                print('')
        elif choice == '2':
            SxY, Sxx, SYY, SSR, Rsq, A, B = generateValues(x,y, True)
            generatePlot(A,B,x,y)
        elif choice == '0':
            x,y = getValues([],[])
            generateValues(x,y, True)
if __name__ == "__main__":
    menu(x,y)
    
    while Flag == True:
        print("The lengths of x and y are not equal. Please reinput values.")
        x,y = getValues(x,y)

