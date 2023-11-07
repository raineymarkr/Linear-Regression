import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r'C:\Users\raine\Downloads\bottle.csv')
depth_df = df[df['Depthm'] <= 10]

temp_df = depth_df['T_degC']
salinity_df = depth_df['Salnty']

combined_df = depth_df[['T_degC', 'Salnty']].dropna()

x = []
y = []

x.extend(combined_df['T_degC'].tolist())
y.extend(combined_df['Salnty'].tolist())

running = False

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

x = np.array(x)
y = np.array(y)

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

yReg = A + B * x

# Graph

print('SxY: ', SxY)
print('Sxx: ', Sxx)
print('SYY: ', SYY)
print('SSR: ', SSR)

print('A: ', A)
print('B: ', B)

plt.scatter(x,y)
plt.plot(x, yReg, color='red')
plt.show()

