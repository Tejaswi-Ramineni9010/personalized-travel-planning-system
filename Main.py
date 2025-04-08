from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import dot
from numpy.linalg import norm
import math
import operator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

main = Tk()
main.title("Personalized Travel Planning System")
main.geometry("1300x1200")

global filename
global X, Y
global user_db, content_db
global vector


def uploadDataset():    
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,filename+" loaded\n")
    
def processDataset():
    text.delete('1.0', END)
    global filename
    global user_db, content_db,merged_df
    user_db = pd.read_csv("Dataset/User.csv",usecols=['Age','Sex','category','Places'])
    content_db = pd.read_csv("Dataset/data_content.csv")
    user_db.fillna(0, inplace = True)
    content_db.fillna(0, inplace = True)
    merged_df = pd.merge(user_db, content_db, on='category', how='inner')
    print("merge df--->",merged_df.head())

    text.insert(END,str(user_db.head())+"\n\n")
    text.insert(END,str(content_db.head())+"\n\n")
    content_db = content_db.values
    user_db = user_db.values
    
def collaborativeModel():
    global X, Y, user_db, content_db
    text.delete('1.0', END)
    X = []
    Y = []
    for i in range(len(user_db)):
        age = str(user_db[i,0]).strip()
        sex = user_db[i,1].strip().lower()
        category = user_db[i,2].strip().lower()
        places = user_db[i,3].strip().lower()
        content = age+" "+sex+" "+category+" "+places
        X.append(content)
        Y.append(category+","+places)
    text.insert(END,"Model generated")

def train_split_test():
    global X_train,y_train,X_test,y_test,merged_df
    label_encoder = LabelEncoder()
    
    merged_df['Sex'] = label_encoder.fit_transform(merged_df['Sex'])
    merged_df['category'] = label_encoder.fit_transform(merged_df['category'])
    merged_df['distance'] = label_encoder.fit_transform(merged_df['distance'])
    merged_df['duration'] = label_encoder.fit_transform(merged_df['duration'])

    X = merged_df[['category', 'distance', 'duration', 'p_rating', 'count']]  # Features
    y = merged_df['itemId']  # Target variable


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    text.insert(END,"Dataset Length : "+str(len(X))+"\n")
    text.insert(END,"Splitted Training Length : "+str(len(X_train))+"\n")
    text.insert(END,"Splitted Test Length : "+str(len(X_test))+"\n\n")                     


def trainRandomForestClassifier():
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_predictions = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print("Random Forest Accuracy:", rf_accuracy)
    text.insert(END,"Random Forest Accuracy : "+str(rf_accuracy)+"\n")




def trainDecisionTreeClassifier():
    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_predictions = dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_predictions)
    print("Decision Tree Accuracy:", dt_accuracy)
    text.insert(END,"Decision Tree Accuracy : "+str(dt_accuracy)+"\n")





def trainKNN():
    global X, Y, vector
    vector = TfidfVectorizer()
    X = vector.fit_transform(X).toarray()
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    print("KNN  Accuracy:", knn_accuracy)
    text.insert(END,"KNN  Accuracy : "+str(knn_accuracy)+"\n")
    text.insert(END,"KNN trained on below dataset vector\n\n")
    text.insert(END,str(X)+"\n\n")

def predict():
    text.delete('1.0', END)
    user_recommend = []
    global X, Y, vector, content_db

    query = tf1.get().lower()
    testArray = vector.transform([query]).toarray()
    testArray = testArray[0]
    for i in range(len(X)):
        recommend = dot(X[i], testArray)/(norm(X[i])*norm(testArray))
        if recommend > 0:
            user_recommend.append([Y[i],recommend])
    user_recommend.sort(key = operator.itemgetter(1),reverse=True)
    top_recommend = []
    for index in range(0,5):
        top_recommend.append(user_recommend[index][0])
    top = max(top_recommend,key=top_recommend.count)
    arr = top.split(",")
    text.insert(END,"Recommended Tourist Destination : "+str(arr[1])+"\n\n")
    text.insert(END,"Below are the nearby places of recommended destination\n\n")
    for i in range(len(content_db)):
        if arr[0] == str(content_db[i,0]).strip().lower():
            distance = str(content_db[i,1]).strip()
            duration = str(content_db[i,2]).strip()
            nearby = str(content_db[i,4]).strip()
            rating = str(content_db[i,6]).strip()
            text.insert(END,"Distance = "+distance+"\n")
            text.insert(END,"Duration = "+duration+"\n")
            text.insert(END,"Nearby Places = "+nearby+"\n")
            text.insert(END,"Rating = "+rating+"\n\n")
    
    
font = ('times', 15, 'bold')
title = Label(main, text='Personalized Travel Planning System')
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Travel Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=20,y=150)
processButton.config(font=ff)

buildButton = Button(main, text="Build Collaborative & Clustering Model", command=collaborativeModel)
buildButton.place(x=20,y=200)
buildButton.config(font=ff)


trainButton = Button(main, text="Train Test Split", command=train_split_test)
trainButton.place(x=20,y=250)
trainButton.config(font=ff)

knnButton = Button(main, text="Train KNN Algorithm", command=trainKNN)
knnButton.place(x=20,y=300)
knnButton.config(font=ff)

rfButton = Button(main, text="Train RF Algorithm", command=trainRandomForestClassifier)
rfButton.place(x=20,y=350)
rfButton.config(font=ff)

dtButton = Button(main, text="Train DT Algorithm", command=trainDecisionTreeClassifier)
dtButton.place(x=20,y=400)
dtButton.config(font=ff)

l1 = Label(main, text='Input Your Requirements')
l1.config(font=font1)
l1.place(x=20,y=450)

tf1 = Entry(main,width=650)
tf1.config(font=font1)
tf1.place(x=20,y=500)

predictButton = Button(main, text="Predict Recommendation", command=predict)
predictButton.place(x=20,y=550)
predictButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=85)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=550,y=100)
text.config(font=font1)

main.config()
main.mainloop()
