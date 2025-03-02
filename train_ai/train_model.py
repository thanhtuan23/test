# import libraries
import numpy as np
import pandas as pd
import os, re, time, math, tqdm, itertools
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as pyo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import keras
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, Dense
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint

# Load all the data in folder dataset
data_folder = 'dataset'

if not os.path.exists(data_folder):
    print(f"Error: Folder {data_folder} does not exist!")
elif not os.listdir(data_folder):
    print(f"Error: Folder {data_folder} is empty!")
else:
    print("Folder exists and has files.")

all_files = []
for root, dirs, files in os.walk(data_folder):
    for file in files:
        all_files.append(os.path.join(root, file))

print(f"Found {len(all_files)} files in {data_folder}")

# load the data into memory
network_data = pd.read_csv(all_files[0])
network_data.head()

# Data Preprocessing
# check for missing values
network_data.isnull().sum()

#For making a proper undertanding of dataset we are using, we will perform a bief EDA (Exploratory Data Analysis) on the dataset
#check shape of the data
network_data.shape

# check the number of rows and columns
print('Number of Rows (Samples): %s' % str((network_data.shape[0])))
print('Number of Columns (Features): %s' % str((network_data.shape[1])))

network_data.head(4)

# check the columns in data
network_data.columns

# check the number of columns
print('Total columns in our data: %s' % str(len(network_data.columns)))

network_data.info()

# check the number of values for labels
network_data['Label'].value_counts()

#Data Visualizations
#After getting some useful information about our data, we now make visuals of our data to see how the trend in our 
# data goes like. The visuals include bar plots, distribution plots, scatter plots, etc.
# plot the number of values for labels
sns.set(rc={'figure.figsize':(12, 6)})
plt.xlabel('Attack Type')
sns.set_theme()
ax = sns.countplot(x='Label', data=network_data)
ax.set(xlabel='Attack Type', ylabel='Number of Attacks')
plt.show()
plt.savefig("output_plot.png")

# make a scatterplot
# pyo.init_notebook_mode()
fig = px.scatter(x = network_data["Bwd Pkts/s"][:100000], 
                 y=network_data["Fwd Seg Size Min"][:100000])
fig

# make a scatterplot
sns.set(rc={'figure.figsize':(12, 6)})
sns.scatterplot(x=network_data['Bwd Pkts/s'][:50000], y=network_data['Fwd Seg Size Min'][:50000], 
                hue='Label', data=network_data)

# check the dtype of timestamp column
(network_data['Timestamp'].dtype)


#Data Preprocessing
# check for some null or missing values in our dataset
network_data.isna().sum().to_numpy()

# drop null or missing columns
cleaned_data = network_data.dropna()
cleaned_data.isna().sum().to_numpy()


#Label Encoding
# encode the column labels
label_encoder = LabelEncoder()
cleaned_data['Label']= label_encoder.fit_transform(cleaned_data['Label'])
cleaned_data['Label'].unique()

# check for encoded labels
cleaned_data['Label'].value_counts()



#Shaping the data for CNN

# make 3 seperate datasets for 3 feature labels
data_1 = cleaned_data[cleaned_data['Label'] == 0]
data_2 = cleaned_data[cleaned_data['Label'] == 1]
data_3 = cleaned_data[cleaned_data['Label'] == 2]

# make benign feature
y_1 = np.zeros(data_1.shape[0])
y_benign = pd.DataFrame(y_1)

# make bruteforce feature
y_2 = np.ones(data_2.shape[0])
y_bf = pd.DataFrame(y_2)

# make bruteforceSSH feature
y_3 = np.full(data_3.shape[0], 2)
y_ssh = pd.DataFrame(y_3)

# merging the original dataframe
X = pd.concat([data_1, data_2, data_3])
y = pd.concat([y_benign, y_bf, y_ssh])

y_1, y_2, y_3

print(X.shape)
print(y.shape)

# checking if there are some null values in data
X.isnull().sum().to_numpy()

#Data Argumentation
from sklearn.utils import resample

data_1_resample = resample(data_1, n_samples=20000, 
                           random_state=123, replace=True)
data_2_resample = resample(data_2, n_samples=20000, 
                           random_state=123, replace=True)
data_3_resample = resample(data_3, n_samples=20000, 
                           random_state=123, replace=True)


train_dataset = pd.concat([data_1_resample, data_2_resample, data_3_resample])
train_dataset.head(2)


# viewing the distribution of intrusion attacks in our dataset 
plt.figure(figsize=(10, 8))
circle = plt.Circle((0, 0), 0.7, color='white')
plt.title('Intrusion Attack Type Distribution')
plt.pie(train_dataset['Label'].value_counts(), labels=['Benign', 'BF', 'BF-SSH'], colors=['blue', 'magenta', 'cyan'])
p = plt.gcf()
p.gca().add_artist(circle)


#Making X & Y Variables (CNN)
test_dataset = train_dataset.sample(frac=0.1)
target_train = train_dataset['Label']
target_test = test_dataset['Label']
target_train.unique(), target_test.unique()

y_train = to_categorical(target_train, num_classes=3)
y_test = to_categorical(target_test, num_classes=3)


#Data Splicing
train_dataset = train_dataset.drop(columns = ["Timestamp", "Protocol","PSH Flag Cnt","Init Fwd Win Byts","Flow Byts/s","Flow Pkts/s", "Label"], axis=1)
test_dataset = test_dataset.drop(columns = ["Timestamp", "Protocol","PSH Flag Cnt","Init Fwd Win Byts","Flow Byts/s","Flow Pkts/s", "Label"], axis=1)

# making train & test splits
X_train = train_dataset.iloc[:, :-1].values
X_test = test_dataset.iloc[:, :-1].values
X_test

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# reshape the data for CNN
X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)
X_train.shape, X_test.shape


# making the deep learning function
def model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', 
                    padding='same', input_shape=(72, 1)))
    model.add(BatchNormalization())
    
    # adding a pooling layer
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', 
                    padding='same', input_shape=(72, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', 
                    padding='same', input_shape=(72, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = model()
model.summary()

checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True)
his = model.fit(X_train, y_train, epochs=30, batch_size=32, 
                validation_data=(X_test, y_test), callbacks=[checkpoint])



#Visualization of Results (CNN)

# check the model performance on test data
scores = model.evaluate(X_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# check history of model
history = his.history
history.keys()

epochs = range(1, len(history['loss']) + 1)
acc = history['accuracy']
loss = history['loss']
val_acc = history['val_accuracy']
val_loss = history['val_loss']

# visualize training and val accuracy
plt.figure(figsize=(10, 5))
plt.title('Training and Validation Accuracy (CNN)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(epochs, acc, label='accuracy')
plt.plot(epochs, val_acc, label='val_acc')
plt.legend()

# visualize train and val loss
plt.figure(figsize=(10, 5))
plt.title('Training and Validation Loss(CNN)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epochs, loss, label='loss', color='g')
plt.plot(epochs, val_loss, label='val_loss', color='r')
plt.legend()





