from collections import Counter
import pandas as pd
import numpy as np
from mpi4py import MPI
import random 
import time

start_time = time.time()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

df = pd.read_csv('out.csv')
df.drop("Unnamed: 0", axis=1, inplace=True)
cls = list(df.columns)
lcls = len(cls) - 1
ratio = (len(cls)-1) // 4

def fit(data,classname,alpha,k):
        pm={}
        parametric_table={}
        yy=alpha*k
        columns=list(data.columns)
        columns.remove(classname)
        for i in columns:
            pm[i]=Counter(data[i])
        class_names=list(Counter(data[classname]).keys())
        for i in pm:
            temp={}
            for j in pm[i].keys():
                for k in class_names:
                    a=data[i]==j
                    b=data[classname]==k
                    c=np.logical_and(a,b).sum()
                    xx=np.array(b).sum()
                    temp[j,k]=(c+alpha)/(xx+yy)
            parametric_table[i]=temp    
        return parametric_table

def predict(q,parametric_table,columns,class_names):
        res=[]
        for i in class_names:
            t=1
            for j in range(len(q)):
                t=t*parametric_table[columns[j]][(q[j],i)]
            res.append(t)
        res=np.array(res)
        return class_names[np.argmax(res)] 


if rank == 0:
    temp=[]
    for i in range(4):
        temp.append(cls[i*ratio:(i+1)*ratio])
    v2 = comm.sendrecv(df[temp[1]+["Class"]],dest=1,source=1)
    v3 = comm.sendrecv(df[temp[2]+["Class"]],dest=2,source=2)
    v4 = comm.sendrecv(df[temp[3]+["Class"]],dest=3,source=3)

    V = {}
    v1 = res = fit(df[temp[0]+["Class"]],"Class",1,lcls)
    for d in [v1, v2, v3, v4]:
        V.update(d)
    #print(V)

    end_time = time.time()
    results = predict(["Weekday","Winter","High","Heavy"],V,cls,list(set(df["Class"])))
    print(results)
    print("Time taken: ", end_time - start_time)

if rank == 1:
    d2 = comm.recv(source=0)
    res = fit(d2,"Class",1,lcls)
    comm.send(res,dest=0)

if rank == 2:
    d3 = comm.recv(source=0)
    res = fit(d3,"Class",1,lcls)
    comm.send(res,dest=0)

if rank == 3:
    d4 = comm.recv(source=0)
    res = fit(d4,"Class",1,lcls)
    comm.send(res,dest=0)