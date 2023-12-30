import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sn
from sklearn.metrics import classification_report

Model = pd.read_excel("/content/drive/MyDrive/Model_Output.xlsx", index_col =[0])

Model.head()

Confusion_Model = pd.crosstab(Model.DNI_Estimation,Model.MS57)

pd.crosstab(Model.DNI_Estimation,Model.MS57)

fig = plt.figure(figsize =(17,5))
ax1 = plt.subplot(121)
sn.heatmap(Confusion_Model,annot=True,cmap='Blues',fmt='g')
ax1.set_title("DNI Estimation Model")

np.diag(Confusion_Model).sum()

Confusion_Model.sum()

Confusion_Model.sum().sum()

np.diag(Confusion_Model).sum()/Confusion_Model.sum().sum()

TP = Confusion_Model.iloc[0,0]
FP = Confusion_Model.iloc[0,:].sum()-TP
FN = Confusion_Model.iloc[:,0].sum()-TP
TN = Confusion_Model.sum().sum()-TP-FP-FN
Accuracy = (TP+TN)/Confusion_Model.sum().sum()
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = (2*Precision*Recall/(Precision+Recall))
TNR = TN/(TN+FP)
FPR = FP/(FP+TN)
FNR = FN/(FN+TP)
MCC = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

for i in range (Confusion_Model.shape[0]):
  TP = Confusion_Model.iloc[i,i]
  FP = Confusion_Model.iloc[i,:].sum()-TP
  FN = Confusion_Model.iloc[:,i].sum()-TP
  TN = Confusion_Model.sum().sum()-TP-FP-FN
  Accuracy = (TP+TN)/(TP+TN+FP+FN)
  
  Recall = TP/(TP+FN)
  F1_Score = (2*Precision*Recall/(Precision+Recall))
  TNR = TN/(TN+FP)
  FPR = FP/(FP+TN)
  FNR = FN/(FN+TP)
  MCC = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
  print(Confusion_Model.index[i],Accuracy,Precision,Recall,F1_Score,TNR,FPR,FNR,MCC)

pd.DataFrame(classification_report(Model.DNI_Estimation,Model.MS57,output_dict=True)).T

results = {}

metric = "ACC"
results[metric] = (TP + TN) / (TP + TN + FP + FN)
print(f"{metric} is {results[metric]: .3f}")

# Sensitivity or Recall
metric = "TPR"
results[metric] = TP / (TP + FN)
print(f"{metric} is {results[metric]: .3f}")

# Specificity
metric = "TNR"
results[metric] = TN / (TN + FP)
print(f"{metric} is {results[metric]: .3f}")
