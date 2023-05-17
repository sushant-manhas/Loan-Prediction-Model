import numpy as np
import pickle
import streamlit as st

# Loading the Model
loaded_model = pickle.load(open('loan_model.sav', 'rb'))

def loan_Prediction(input_values):
    
    
    input_array = np.asanyarray(input_values)
    input_array_reshaped = input_array.reshape(1,-1)
    answer = loaded_model.predict(input_array_reshaped)  # It's a list and not an Integer
    if (answer[0] == 0):
        return 'The Person is Not eligible for Loan'
    else:
        return 'The Person is Eligible for Loan'
    
def main():
    
    st.title('Loan Eligibility Prediction Web App')
    
    # Taking Inputs 
    Gender = st.text_input('Gender (Male: 1, Female: 0)')
    Married = st.text_input('Marital Status (Married: 1, Not-Married: 0)')
    Dependents = st.text_input('Number of Dependents')
    Education = st.text_input('Education Level (Graduated: 1, Not-Graduated: 0)')
    Self_Employed = st.text_input('Self Employed or not (Y: 1, N: 0)')
    ApplicantIncome = st.text_input('Income of the Applicant')
    CoapplicantIncome = st.text_input('Income of the Co-Applicant (0 if no co-applicant)')
    LoanAmount = st.text_input('Amount of Loan to be Taken ()')
    Loan_Amount_Term = st.text_input('Term for Loan Amount to be Taken ()')
    Credit_History = st.text_input('Earlier Credit History (Y: 1, N: 0)')
    Property_Area = st.text_input('Area in which Property owned (Rural:0, Semiurban:1, Urban:2)')
    
    eligibility = ''  # A Null String to store final Value
    
    if st.button('Predict Eligibility'):
        eligibility = loan_Prediction([Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area])
        
        
    st.success(eligibility)
    
if __name__ == '__main__':
    main()
