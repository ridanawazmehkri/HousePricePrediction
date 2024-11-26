from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np

app=Flask(__name__)
data=pd.read_csv('Cleaned_data.csv')

pipe=pickle.load(open("LinearRegression.pkl",'rb'))

@app.route('/')
def index():

   locations=data['location'].unique()
   return render_template('index.html',locations=locations)


@app.route('/predict',methods=['POST'])
def predict():
   location=request.form.get('location')
   bhk = request.form.get('BHK')
   bath=request.form.get('Bath')
   sqft=request.form.get('total_sqft')

   input=pd.DataFrame([[location,sqft,float(bath),float(bhk)]],columns=['location','total_sqft','bath','BHK'])
   prediction=pipe.predict(input)[0] * 100000
   
   return " "+str(np.round(prediction,2))


if __name__=="__main__":
   app.run(debug=True,port=5001)
   
