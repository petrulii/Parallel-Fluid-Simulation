import os
import matplotlib.pyplot as plt
'''
X = [2,4,8,16]
plt.figure()
for i in range(1,4):
        Y=[]
        for j in X:
                cmd = "mpirun --use-hwthread-cpus -np "+str(j)+" ./lbm -e "+str(i)+" -n"
                sumT=0
                for h in range(0,10):
                        print(i,j,h)
                        stream = os.popen(cmd)
                        output = stream.read()
                        splited = output.split("\n")
                        time = splited[-2].split(" ")[2]
                        sumT=sumT+float(time)
                Y.append((sumT/10))
        plt.plot(X, Y, '.-' ,  label="Ex "+str(i))
plt.legend(loc="upper left")
plt.savefig('measurements123')
plt.show()'''
plt.figure()
for i in range(6,7):
        Y=[]
        for j in X:
                cmd = "mpirun --use-hwthread-cpus -np "+str(j)+" ./lbm -e "+str(i)+" -n"
                sumT=0
                for h in range(0,10):
                        print(i,j,h)
                        stream = os.popen(cmd)
                        output = stream.read()
                        splited = output.split("\n")
                        time = splited[-2].split(" ")[2]
                        sumT=sumT+float(time)
                Y.append((sumT/10))
        plt.plot(X, Y, '.-' ,  label="Ex "+str(i))
plt.legend(loc="upper left")
plt.savefig('measurements456')
plt.show()