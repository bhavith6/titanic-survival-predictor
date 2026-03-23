import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('train.csv')
median_age=df['Age'].median()
df['Age']=df['Age'].fillna(median_age)
print(f'the missing values are:{df["Age"].isnull().sum()} ')
print(median_age)
target=np.array(df['Survived'])
df['Sex']=df['Sex'].replace({'male':0,'female':1})
para_1=np.array(df[['Age','Sex','Pclass']])
weights=np.random.rand(3)
bias=np.random.rand(1)
log_error=[]
learning_rate=0.001
flag=0
log_error=[]
epsilon=1e-9
count=0
while(flag==0):
    prediction=np.dot(para_1,weights)+bias
    prediction=1/(1+np.exp(-prediction))
    weights=weights+learning_rate*(np.dot(target-prediction,para_1)/len(target))
    bias=bias+learning_rate*(np.sum(target-prediction)/len(target))
    likelyhood=-np.mean(target*np.log(prediction+epsilon)+(1-target)*np.log(1-prediction+epsilon))
    log_error.append(likelyhood)
    count+=1
    if(count>1):
        diffrence=abs(log_error[-1]-log_error[-2])
        if(diffrence<=0.000001):
            print("-----tranining_done-----")
            flag=1
print(f"the weights are {weights}")  
print(f"the bias is{bias}") 
plt.plot(log_error)
plt.title("error_history")
plt.show()
x=int(input("enter the number people you want to check: "))
for i in range(x):
    a=int(input("enter the your age lets see you gonna alvie or dead if you are at titanic :"))
    b=int(input("enter the your sex lets see you gonna alvie or dead if you are at titanic :"))
    c=int(input("enter the class entered in titnaic"))
    x=a*weights[0]+b*weights[1]+c*weights[2]+bias
    print(f"the chances of survivial rate is: {(1/(1+np.exp(-x)))*100}%")
np.savez('titanic_model.npz',saved_weights=weights,saved_bias=bias)
print("susscefully saved!")
