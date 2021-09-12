import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd

st.set_page_config(
    page_title="Student Progress Tracker",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Student Tracker")
st.image("assets\img_kys.png", width=300)

grade_name = st.sidebar.selectbox("Select grade",("4","5","6","7","8","9","10"))
roll_no = st.sidebar.slider("Select Roll No",1,120)
df=pd.read_csv('assets/Dataset/xAPI-Edu-Data.csv')


X = df.drop(columns='Class')
y = df['Class']

X = pd.get_dummies(X)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=42)
model = MLPClassifier(random_state=42, max_iter=2000)
model.fit(train_X, train_y)
predictions = model.predict(test_X)



def predict_progress(grade_name,j):
  j = roll_no
  st.write("## THE DATA BEING USED")
  Xs = X
  grade='GradeID_G-0'+str(grade_name)
  Xs.loc[Xs[grade] == 1]
  ip=Xs[j-1:j]
  l=""
  st.write("## CHOSEN WEIGHTINGS: ")
  ip
  op=model.predict(ip)
  op_list = op.tolist()
  if(op_list[0]=='M'):
    stat= "Above average"
  elif(op_list[0]=='L'):
    stat = "Below average"
  elif(op_list[0]=='H'):
    stat = "Good"
  exp ="Student performance : {} ".format(stat) 
  l=l+exp
  if(stat=="Below average"):
    if(X[j-1:j]['raisedhands'].tolist()[0]<40):
      r1="Not interacting"
      rs1="Reasons found : {}"  .format(r1)
      l=l+rs1
    if(X[j-1:j]['StudentAbsenceDays_Under-7'].tolist()[0]==1):
      r2="Less attendance"
      rs2=", {} " .format(r2)
      l=l+rs2
    if(X[j-1:j]['AnnouncementsView'].tolist()[0]<40):
      r3="Not viewing class announcements regularly"
      rs3=", {} " .format(r3)
      l=l+rs3
      st.write("## PREDICTION OUTPUT: ")
  return l

st.header(predict_progress(grade_name,roll_no))
