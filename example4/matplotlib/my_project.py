import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np

with open("../../../private_nccl_tests/performance") as file_in:
    xdata1 = []
    xdata2 = []
    ydata1 = []
    ydata2 = []

    column0 = []
    column1 = []

    columns = file_in.read().split("----------------------------------\n")
    columns.pop(-1)

    column0 = columns[0].split("\n")
    column1 = columns[1].split("\n")

    if "" in column0:
        column0.remove("")

    if "" in column1:
        column1.remove("")


    #we have two arrays of numbers here

    for line in column0:
        numbers = line.split(",")
        x = float(numbers[0])
        xdata1.append(x)
        y = float(numbers[1])
        ydata1.append(y)
        #print ("x = %15f           y = %15f" %(x,y))
        #print (line.strip())
        #print (float(line.strip()))

    print ("\n\n")

    for line in column1:
        numbers = line.split(",")
        x = float(numbers[0])
        xdata2.append(x)
        y = float(numbers[1])
        ydata2.append(y)
        #print ("x = %15f           y = %15f" %(x,y))
        #print (line.strip())
        #print (float(line.strip()))

    #for x in xdata1: 
    #   print ("%15f\n" %(x))
    #for y in ydata2: 
    #   print ("%15f\n" %(y))


    
    #print ("\n\n the length is % 2d" %(len(column0)))
    #print ("\n\n the length is % 2d" %(len(column1)))



    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xdata1, ydata1, color='tab:blue')
    ax.plot(xdata2, ydata2, color='tab:orange')
    plt.show()
