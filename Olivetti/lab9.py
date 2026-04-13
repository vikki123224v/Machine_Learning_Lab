import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
data=fetch_olivetti_faces(shuffle=True,random_state=42)
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(f'Accuracy:{accuracy*100:.2f}%')
print("\nClassification Report:")
print(classification_report(y_test,y_pred,zero_division=1))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,y_pred))

fig,axes=plt.subplots(3,5,figsize=(12,8))
for ax,image,label,prediction in zip (axes.ravel(),x_test,y_test,y_pred):
    ax.imshow(image.reshape(64,64),cmap=plt.cm.gray)
    ax.set_title(f"True:{label}pred:{prediction}")
    ax.axis('off')
plt.tight_layout()
plt.show()