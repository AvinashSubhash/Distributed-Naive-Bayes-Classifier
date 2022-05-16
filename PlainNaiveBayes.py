# create navie bayes classifier for categorical dataset from scratch

import pandas as pd
import numpy as np
from collections import Counter
import time

start_time = time.time()

class NaiveBayes:
    def __init__(self):
        self.parametric_table={}
        
    def fit(self,data,classname,alpha,k):
        self.pm={}
        yy=alpha*k
        self.columns=list(data.columns)
        self.columns.remove(classname)
        for i in self.columns:
            self.pm[i]=Counter(data[i])
        self.class_names=list(Counter(data[classname]).keys())
        for i in self.pm:
            temp={}
            for j in self.pm[i].keys():
                for k in self.class_names:
                    a=data[i]==j
                    b=data[classname]==k
                    c=np.logical_and(a,b).sum()
                    xx=np.array(b).sum()
                    temp[j,k]=(c+alpha)/(xx+yy)
            self.parametric_table[i]=temp    
        return self.parametric_table
    
    def predict(self,q):
        res=[]
        for i in self.class_names:
            t=1
            for j in range(len(q)):
                t=t*self.parametric_table[self.columns[j]][(q[j],i)]
            res.append(t)
        res=np.array(res)
        return self.class_names[np.argmax(res)] 

N = NaiveBayes()

air_traffic = pd.read_csv('out.csv')
# x = data[["Days","Season","Fog","Rain"]]
# y = data["Class"]
#print(air_traffic.head)
air_traffic = air_traffic.drop(["Unnamed: 0"], axis=1)
N.fit(data=air_traffic,classname="Class",alpha=1,k=4)
end_time = time.time()
res=N.predict(["Weekday","Winter","High","Heavy"])
print(res)
print("Time taken: ", end_time - start_time)


