# STEP 1: Import dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
print("✅ Libraries are installed and working!")

# STEP 2: Data Collection and Analysis 
#2.1 Loading the diabetes dataset to a pandas DataFrame
# Download the dataset from Kaggle: https://www.kaggle.com/uciml/pima-indians-diabetes-database
import kagglehub
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
diabetes_dataset = pd.read_csv(path + "/diabetes.csv")  # reading the file inside that dataset is diabetes.csv
print("First 5 rows of the dataset:") #checking if data is loaded properly
print(diabetes_dataset.head())

# 2.2 Number of Rows and Columns in this dataset
print("Dataset shape (rows, columns):", diabetes_dataset.shape)

# 2.3 Getting the statistical measures of the data 
print("Statistical measures of the dataset:", diabetes_dataset.describe()) 

# 2.4 How many cases are there for diabetes and non-diabetes
print(diabetes_dataset['Outcome'].value_counts()) #Label 0 --> non-diabetic, Label 1 --> diabetic

# 2.5 Finding the mean by outcome
print("Mean of each column in the dataset grouped by Outcome:", diabetes_dataset.groupby('Outcome').mean())
#good practice to group datasets based on labels 

# STEP 3: Data Preprocessing
#Separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1) #dropping the column Outcome from the dataset
Y = diabetes_dataset['Outcome'] 

#3.1 Data Standardization: why? so that the model can make better predictions
scaler = StandardScaler() 
scaler.fit(X)
standardized_data = scaler.transform(X)
print("Standardized data:", standardized_data)

X = standardized_data #represents data
Y = diabetes_dataset['Outcome'] #represents the labels 
print("X:", X)
print("Y:", Y)

# STEP 4: Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2) #giving the model test data of 20%
print("Total data:", X.shape, "Training data:", X_train.shape, "Test data:", X_test.shape) 

#STEP 5: Training the Model 
classifier = svm.SVC(kernel='linear') #using support vector machine classifier
classifier.fit(X_train, Y_train)
print("✅ Model is trained!")

# STEP 6: Model Evaluation
# 6. 1 Accuracy Score of the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) #using the model to predict the labels, comparing with the actual labels (Y_train), giving us the accuracy score 
print("Accuracy score of the training data:", training_data_accuracy) #above 75% is good! 

# 6. 2 Accuracy Score of the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test) #using the model to predict the labels, comparing with the actual labels (Y_test), giving us the accuracy score
print("Accuracy score of the test data:", test_data_accuracy) 

# STEP 7: Making a Predictive System
input_data = (4,110,92,0,0,37.6,0.191,30) #example input data from csv, and model has to predict either 0 (non-diabetic) or 1 (diabetic) correcly (should be 0)

#7.1 Changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data) 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) #reshaping the data as we are predicting for ONE data point, not the entire dataset

#7.2 Standardizing the input data, same manner we standardized the entire dataset
std_data=scaler.transform(input_data_reshaped)
print("Standardized input data:", std_data)

#7.3 Making the prediction
prediction = classifier.predict(std_data) #making the outcome prediction
print("Prediction:", prediction) #0 or 1
if (prediction[0] == 0): #we have to give an index because the prediction is a list, not an int. the list has only one element so we use [], we get the first value in the list
    print("The person is not diabetic")
else:
    print("The person is diabetic")

# END OF PROJECT
