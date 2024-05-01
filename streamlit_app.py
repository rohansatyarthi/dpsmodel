import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset and perform data cleaning
df = pd.read_csv('diabetes (5).csv')

# Remove duplicates
df = df.drop_duplicates()

# Replace zeros with mean in specific columns
cols_to_replace_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_replace_zeros:
    df[col] = df[col].replace(0, df[col].mean())

# Preprocess the data (handle missing values, scale features, etc.)

# This step includes splitting the dataset into features (X) and target (y)

# Assuming X contains features and y contains target variable
# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['Outcome'])  # Assuming 'Outcome' is the name of your target variable
y = df['Outcome']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train your models
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

svm = SVC()
svm.fit(X_train, y_train)

nb = GaussianNB()
nb.fit(X_train, y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

def predict_diabetes(X):
    predictions = []
    
    # Predict whether person is diabetic or not using each model
    predictions.append(1 if log_reg.predict(X)[0] == 1 else 0)
    predictions.append(1 if rf.predict(X)[0] == 1 else 0)
    predictions.append(1 if svm.predict(X)[0] == 1 else 0)
    predictions.append(1 if nb.predict(X)[0] == 1 else 0)
    predictions.append(1 if knn.predict(X)[0] == 1 else 0)
    
    # Calculate average percentage of being diabetic
    avg_percentage = sum(predictions) / len(predictions) * 100
    
    # Get individual results for each algorithm
    individual_results = {
        "Logistic Regression": "Person is Diabetic" if log_reg.predict(X)[0] == 1 else "Person is Not Diabetic",
        "Random Forest": "Person is Diabetic" if rf.predict(X)[0] == 1 else "Person is Not Diabetic",
        "SVM": "Person is Diabetic" if svm.predict(X)[0] == 1 else "Person is Not Diabetic",
        "Naive Bayes": "Person is Diabetic" if nb.predict(X)[0] == 1 else "Person is Not Diabetic",
        "KNN": "Person is Diabetic" if knn.predict(X)[0] == 1 else "Person is Not Diabetic"
    }
    
    return avg_percentage, individual_results

st.title("Diabetes Prediction System")

# Create input fields
with st.form(key='input_form'):
    st.subheader("Enter Patient Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input('Number of Pregnancies')
        skin_thickness = st.number_input('Skin Thickness Value')
        dpf = st.number_input('Diabetes Pedigree Function value')

    with col2:
        glucose = st.number_input('Glucose Level')
        insulin = st.number_input('Insulin Level')
        age = st.number_input('Age of the person')

    with col3:
        blood_pressure = st.number_input('Blood Pressure Value')
        bmi = st.number_input('BMI Value')

    predict_button = st.form_submit_button(label='Predict')

# Perform prediction when predict button is clicked
if predict_button:
    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
    avg_result, individual_results = predict_diabetes(input_data)

    # Update result label with average percentage
    st.subheader("Diabetes Test Result")
    if avg_result >= 50:
        st.success(f"Average Probality of Person Being Diabetic: {avg_result:.2f}%")
    else:
        st.error(f"Average Probality of Person Being Diabetic: {avg_result:.2f}%")

    # Display individual results for each algorithm
    st.subheader("Result of Individual Algorithms")
    for algo, result in individual_results.items():
        st.write(f"{algo}: {result}")
