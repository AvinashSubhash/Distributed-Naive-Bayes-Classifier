class NaiveBayes:
    def __init__(self):
        self.parametric_table={}
        
    def fit(self,data,classname):
        self.pm={}
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
                    b=data["Class"]==k
                    c=np.logical_and(a,b).sum()
                    temp[j,k]=c/np.array(b).sum()
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