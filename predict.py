import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
brain=np.load('titanic_model.npz')
weights=brain['saved_weights']
bias=brain['saved_bias']
a=int(input("enter the number of people you want to predict: "))
for i in range(a):
    age=int(input("enter the passanger age: "))
    sex=int(input("enter the passenger sex (Male=0,Feamle=1): "))
    pclass=int(input("enter the class among  1,2,3: "))
    x=age*weights[0]+sex*weights[1]+pclass*weights[2]+bias
    probability=1/(1+np.exp(-x))
    print(f"the change of the survival are: {probability[0]*100:.2f}%")