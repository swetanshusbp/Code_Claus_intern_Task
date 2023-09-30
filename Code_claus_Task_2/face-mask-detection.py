#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


from glob import glob
import matplotlib.pyplot as plt
import cv2
import xml.etree.ElementTree as ET
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split


# In[3]:


masked_images = glob(os.path.join('/kaggle/input/face-mask-detection/images/*.png'))


# In[4]:


len(masked_images)


# In[5]:


masked_annotations = glob(os.path.join('/kaggle/input/face-mask-detection/annotations/*.xml'))


# In[6]:


len(masked_annotations)


# In[7]:


masked_annotations[0]


# In[8]:


annotations_data = pd.DataFrame()

for annotation in masked_annotations:
    tree = ET.parse(annotation)
    root = tree.getroot()
    counter = 0
    dic = dict()
    dic['Image_Name'] = root[1].text
    for i in range(4, len(root)):
        temp = []
        temp.append(root[i][0].text)
        for point in root[i][5]:
            temp.append(point.text)
        dic['wear_mask ' + str(counter)] = temp
        counter += 1

    annotations_data = pd.concat([annotations_data, pd.DataFrame([dic])], ignore_index=True)
    counter = 0


# In[9]:


annotations_data = annotations_data.fillna(5)


# In[10]:


annotations_data.head()


# In[11]:


image_directory = '/kaggle/input/face-mask-detection/images/'
labels = []
data = []
classes = ["without_mask" , "mask_weared_incorrect" , "with_mask"]
labels = []


# In[12]:


for index , row in annotations_data.iterrows():
    img = cv2.imread(os.path.join(image_directory , row[0]))
    cv2.resize(img , (255 , 255))
    for obj in row[1:]:
        if obj != 5:
            label = obj[0]
            obj[0] = obj[0].replace(str(label) , str(classes.index(label)))
            obj=[int(each) for each in obj]
            face = img[obj[2]:obj[4],obj[1]:obj[3]]
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            data.append(face)
            labels.append(label)
            if(label=="mask_weared_incorrect"):
                data.append(face)
                labels.append(label)


# In[13]:


data = np.array(data , dtype="float32")
labels = np.array(labels)


# In[14]:


print(data.shape)
print(labels.shape)


# In[15]:


lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(data, labels,test_size=0.2, stratify=labels, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


# In[17]:


print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)
print(X_test.shape)
print(Y_test.shape)


# In[18]:


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam


# In[19]:


model = Sequential() 

model.add(Flatten(input_shape = (224 , 224 , 3))) 
model.add(Dense(200, activation=LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.5)) 

model.add(Dense(50, activation=LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.3)) 

model.add(Dense(3 , activation = 'softmax'))


# In[20]:


model.summary()


# In[21]:


INIT_LR = 1e-4
EPOCHS = 15
BS = 1


# In[22]:


from keras.optimizers import Adam

# Define the learning rate schedule
def lr_schedule(epoch):
    lr = INIT_LR
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr

# Create a custom Adam optimizer with the learning rate schedule
opt = Adam(learning_rate=INIT_LR)

# Compile the model with the custom optimizer
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# In[23]:


opt = Adam(lr = INIT_LR , decay = INIT_LR / EPOCHS)
model.compile(loss = "categorical_crossentropy" , optimizer = opt , metrics = ["accuracy"])
H = model.fit(X_train, Y_train , steps_per_epoch = len(X_train) // BS , validation_data = (X_val , Y_val) , validation_steps = len(X_val) // BS , epochs = EPOCHS)


# In[25]:


from sklearn.metrics import accuracy_score
from sklearn import metrics
import numpy as np
import seaborn as sns


# In[26]:


y_pred = model.predict(X_test)


# In[27]:


y_pred_new = np.argmax(y_pred , axis = 1) # To get the index (The class numer) of the predicted class
y_test_new = np.argmax(Y_test , axis = 1)


# In[28]:


y_pred_new[:10]


# In[30]:


Confusion_Mtrx = metrics.confusion_matrix(y_test_new , y_pred_new)
#true_positive , false_positive, false_negative, true_negative = Confusion_Mtrx.ravel()

plt.figure(figsize=(6,6))
sns.heatmap(Confusion_Mtrx, annot=True, fmt=".2f");
plt.ylabel('Real Value');
plt.xlabel('Predicted Values');
plt.title('Accuracy {0:.2f}'.format(metrics.accuracy_score(y_test_new, y_pred_new)));


# In[ ]:




