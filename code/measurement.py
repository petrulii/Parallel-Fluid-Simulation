import numpy as np
import os
import matplotlib.pyplot as plt
X = [2,4,8]
plt.figure()
for i in range(1,7):
        Y=[]
        for j in X:
                cmd = "mpirun --use-hwthread-cpus -np "+str(j)+" ./lbm -e "+str(i)+" -n"
                stream = os.popen(cmd)
                output = stream.read()
                splited = output.split("\n")
                time = splited[-2].split(" ")[2]
                Y.append(float(time))
        plt.plot(X, Y, '.-' ,  label="Ex "+str(i))
plt.legend(loc="upper left")
plt.savefig('measurements')
plt.show()