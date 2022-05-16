from collections import Counter
from unittest import result
import pandas as pd
import numpy as np
import multiprocessing as mp
import threading as th
import random
import time
import NaveBayes as nb

k_val=1
df = pd.read_csv('out.csv')
#print(df)
df.drop("Unnamed: 0", axis=1, inplace=True)
cls = list(df.columns)
alpha = len(cls) - 1
ratio = (len(cls)-1) // 4

fit_val=[]
def fit(data,classname,queue):
        pm={}
        parametric_table={}
        yy=alpha*k_val
        columns=list(data.columns)
        columns.remove(classname)
        for i in columns:
            pm[i]=Counter(data[i])
        class_names=list(Counter(data[classname]).keys())
        for i in pm:
            temp={}
            for j in pm[i].keys():

                OuterTh = th.Thread(target=OuterThread,args=(class_names,data,i,j,temp,yy,classname))
                OuterTh.start()
                OuterTh.join()

            parametric_table[i]=temp    
        queue.put(parametric_table)

def predict(q,parametric_table,columns,class_names):
        res=[]
        #print(class_names)
        for i in class_names:
            t=1
            for j in range(len(q)):
                t=t*parametric_table[columns[j]][(q[j],i)]
            res.append(t)
        res=np.array(res)
        return class_names[np.argmax(res)],max(res)

def ThreadedWork(data,i,j,k,classname,temp,yy):
    a = data[i] == j
    b = data[classname] == k
    c = np.logical_and(a,b).sum()
    xx = np.array(b).sum()
    temp[j,k] = (c+alpha)/(xx+yy)

def OuterThread(class_names,data,i,j,temp,yy,classname):
    for k in class_names:
        InnerThread = th.Thread(target=ThreadedWork,args=(data,i,j,k,classname,temp,yy))
        InnerThread.start()

def DivideAndWork(temp,fit_val):
    for i in range(len(temp)):
        p = mp.Process(target=fit,args=(df[temp[i]+["Class"]],"Class",fit_val))
        p.start()


if __name__ == "__main__":
    #print(cls)
    temp=[]
    fit_val = mp.Queue()
    result={}
    a=time.time()
    for i in range(4):
        temp.append(cls[i*ratio:(i+1)*ratio])
    x = mp.Process(target=DivideAndWork,args=(temp,fit_val))
    x.start()
    x.join()
    while not fit_val.empty():
        result.update(fit_val.get())
    b = time.time()
    time_for_parallel = b-a
    #print(result)
    print("Time Difference: ",time_for_parallel)
    
    
    #Menu Driven Section
    option = 1
    while True:
        print("Enter the Class Options to select: ")
        input_list = []
        for m in range(len(cls)):
            if cls[m]!="Class":
                option_list = list(df[cls[m]].unique())
                for gg in range(len(option_list)):
                    print(gg,": ",option_list[gg])
                #print(df[cls[m]].unique())
                print("\n\n")
                f = int(input("Option: "))
                print("\n")
                if f < 0 or f >= len(option_list):
                    exit()
                input_list.append(option_list[f])
        final_output,perc = predict(input_list,result,cls,list(set(df["Class"])))
        print("Result: \n")
        print(final_output, " with Probability: ",perc)
        break

