import numpy as npy
file = open('ex3data1X.csv','rt')
dataX = file.read()
dataXsplit = dataX.split()
dataX = []
for i in range(len(dataXsplit)):
    ele = dataXsplit[i][:].split(',')
    row = []
    for element in ele:
        row.append(float(element))
    dataX.insert(i, row)
file = open('ex3data1y.csv','rt')
dataYstr = file.read()
dataYsplit = dataYstr.split()
dataY = []
for element in dataYsplit:
    dataY.append(int(element))

def trainingExampleProvider(x, y, dataX, dataY):
    x1 = [1]
    y1 = []
    for i in range(20):
        for j in range(20):
            x1.append(dataX[x+i][y+j])
    for i in range(10):
        y1.append(0)
    num = dataY[20*x + y]
    if num == 10:
        y1[0] = 1
    else:
        y1[num] = 1
    return x1, y1

def sigmoid(z):
    return (1/(1 + npy.exp(-z)))

def outputBlackBox(x1, theta1, theta2):
    a2 = [1]
    for i in range(len(theta1)):
        z = 0
        for j in range(len(theta1[0])):
            z += theta1[i][j]*x1[j]
        a2.append(sigmoid(z))
    a3 = []
    for i in range(len(theta2)):
        z = 0
        for j in range(len(theta2[0])):
            z += theta2[i][j]*a2[j]
        a3.append(sigmoid(z))
    return a2, a3

def cost(dataX, dataY, theta1, theta2, regParam):
    m = int(len(dataX)/20)
    n = int(len(dataX[0])/20 - 5)
    cost1 = 0
    for x in range(m):
        for y in range(n):
            x1, y1 = trainingExampleProvider(x, y, dataX, dataY)
            a2, a3 = outputBlackBox(x1, theta1, theta2)
            for z in range(len(a3)):
                cost1 += -(y1[z]*npy.log(a3[z]) + (1 - y1[z])*npy.log(1 - a3[z]))
    cost2 = 0
    for i in range(len(theta1)):
        for j in range(len(theta1[0])):
            cost2 += theta1[i][j]*theta1[i][j]
    for i in range(len(theta2)):
        for j in range(len(theta2[0])):
            cost2 += theta2[i][j]*theta2[i][j]
    cost2 = cost2*regParam/2
    cost = cost1 + cost2
    cost = cost/(m*n)
    return cost

def backPropagation(dataX, dataY, theta1, theta2, regParam, alpha, maxIter):
    m = int(len(dataX)/20)
    n = int(len(dataX[0])/20 - 5)
    for count in range(maxIter):
        grad2 = npy.zeros([len(theta2), len(theta2[0])])
        grad1 = npy.zeros([len(theta1), len(theta1[0])])
        for x in range(m):
            for y in range(n):
                x1, y1 = trainingExampleProvider(x, y, dataX, dataY)
                a2, a3 = outputBlackBox(x1, theta1, theta2)
                del3 = []
                for c1 in range(len(a3)):
                    del3.append(a3[c1] - y1[c1])

                del2 = []
                for c1 in range(len(theta2[0])):#101
                    del2_row = 0
                    for c2 in range(len(theta2)): #10
                        del2_row += del3[c2]*theta2[c2][c1]
                        grad2[c2][c1] += a2[c1]*del3[c2]
                    del2.append(del2_row*a2[c1]*(1-a2[c1]))
                for c1 in range(len(theta1)): #100
                    for c2 in range(len(theta1[0])): #401
                        grad1[c1][c2] += x1[c2]*del2[c1]
        for i in range(len(theta2)):
            for j in range(len(theta2[0])):
                if j==0:
                    grad2[i][j] = grad2[i][j]/(m*n)
                else:
                    grad2[i][j] = grad2[i][j]/(m*n) + regParam*theta2[i][j]
        for i in range(len(theta1)):
            for j in range(len(theta1[0])):
                if j==0:
                    grad1[i][j] = grad1[i][j]/(m*n)
                else:
                    grad1[i][j] = grad1[i][j]/(m*n) + regParam*theta1[i][j]
        for i in range(len(theta2)):
            for j in range(len(theta2[0])):
                theta2[i][j] = theta2[i][j] - alpha*grad2[i][j]
        for i in range(len(theta1)):
            for j in range(len(theta1[0])):
                theta1[i][j] = theta1[i][j] - alpha*grad1[i][j]
        print(theta1)
        print(theta2)
    return theta1, theta2

theta1_initial = npy.random.rand(100, 401)
theta2_initial = npy.random.rand(10, 101)
regParam = 10
alpha = 0.003
maxIter = 500
theta1, theta2 = backPropagation(dataX, dataY, theta1_initial, theta2_initial, regParam, alpha, maxIter)
p = 0
for x in range(int(len(dataX)/20)):
    for y in range(int(len(dataX[0])/20 - 15)):
        x1, y1 = trainingExampleProvider(x, y, dataX, dataY)
        a2, a3 = outputBlackBox(x1, theta1, theta2)
        max = a3[0]
        maxIndex = 0
        for i in range(len(a3)):
            if a3[i]>max:
                max = a3[i]
                maxIndex = i
        if y1[maxIndex] == 1:
            p += 1
accuracy = p*100/((len(dataX)/20)*(len(dataX[0])/20 - 15))
print(accuracy)


           



