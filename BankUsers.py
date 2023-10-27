import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle

model = pickle.load(open('BankUsers.pkl', 'rb'))
st.markdown("<h1 style = 'text-align: center; color: 3D0C11'>BANK USERS PROJECT </h1> ", unsafe_allow_html = True)
st.markdown("<h6 style = 'top_margin: 0rem; text-align: center; color: #FFB4B4'>Built by GoMyCode Sanaith Wizard</h6>", unsafe_allow_html = True)
st.image('pngwing.com (2).png', width = 400)


st.subheader('Project Brief')

st.markdown("<p style = 'top_margin: 0rem; text-align: justify; color: #FFB4B4'> In the dynamic and ever-evolving landscape of entrepreneurship, startups represent the vanguard of innovation and economic growth. The inception of a new venture is often accompanied by great enthusiasm and ambition, as entrepreneurs strive to transform their groundbreaking ideas into successful businesses. However, one of the central challenges faced by startups is the uncertainty surrounding their financial sustainability and profitability. This uncertainty is exacerbated by a myriad of factors,<br> ranging from market volatility and competition to operational costs and customer acquisition.</p>", unsafe_allow_html = True)

st.markdown("<br><br>", unsafe_allow_html = True)

username = st.text_input('Enter your name')
if st.button('submit name'):
    st.success(f"Welcome {username}. Pls use according to usage guidelines")


data = pd.read_csv('Financial_inclusion_dataset.csv')
st.write(data.sample(10))

st.sidebar.image('pngwing.com (4).png', caption= f'Welcome {username}')

input_type = st.sidebar.selectbox('Select Your preffered Input type', ['Slider Input', 'Number Input'])

if input_type == "Slider Input":
    location_type = st.sidebar.select_slider('location_type', data['location_type'].unique())
    cellphone_access = st.sidebar.select_slider('cellphone_access', data['cellphone_access'].unique())
    household_size = st.sidebar.slider("household_size", data['household_size'].min(), data['household_size'].max())
    age_of_respondent = st.sidebar.slider("age_of_respondent", data['age_of_respondent'].min(), data['age_of_respondent'].max())
    gender_of_respondent = st.sidebar.select_slider('gender_of_respondent', data['gender_of_respondent'].unique())
    relationship_with_head = st.sidebar.select_slider('relationship_with_head', data['relationship_with_head'].unique())
    marital_status = st.sidebar.select_slider('marital_status', data['marital_status'].unique())
    education_level = st.sidebar.select_slider('education_level', data['education_level'].unique())
    job_type = st.sidebar.select_slider('job_type', data['job_type'].unique())
else:
    household_size = st.sidebar.number_input("household_size", data['household_size'].min(), data['household_size'].max())
    age_of_respondent = st.sidebar.number_input("age_of_respondent", data['age_of_respondent'].min(), data['age_of_respondent'].max())


input_variable = pd.DataFrame([{"location_type":location_type, "cellphone_access": cellphone_access, "household_size": household_size, "age_of_respondent": age_of_respondent, "gender_of_respondent": gender_of_respondent, "relationship_with_head": relationship_with_head, "marital_status": marital_status, "education_level": education_level, "job_type": job_type }])
st.write(input_variable)


from sklearn.preprocessing import LabelEncoder, StandardScaler
def transformer(dataframe):
    lb = LabelEncoder()
    scaler = StandardScaler()
    #dep = dataframe.drop('bank_account', axis=1)

     # scale the numerical columns
    for i in dataframe:# ---------------------------------------------- Iterate through the dataframe columns
        if i in dataframe.select_dtypes(include = 'number').columns: # --------- Select only the numerical columns
            dataframe[[i]] = scaler.fit_transform(dataframe[[i]]) # ------------ Scale all the numericals

    # label encode the categorical columns
    for i in dataframe.columns:  # --------------------------------------------- Iterate through the dataframe columns
        if i in dataframe.select_dtypes(include = ['object', 'category',]).columns: #-- Select all categorical columns
            dataframe[i] = lb.fit_transform(dataframe[i]) # -------------------- Label encode selected categorical columns
    return dataframe

transformer(input_variable)

#from sklearn.preprocessing import LabelEncoder
#lb = LabelEncoder() 
#for i in input_variable.columns:
#    if input_variable[i].dtypes == 'O':
#        input_variable[i] = lb.fit_transform(input_variable[i])

pred_result, interpret = st.tabs(["Prediction Tab", "Interpretation Tab"])
with pred_result:
    if st.button('PREDICT'):

        st.markdown("<br>", unsafe_allow_html= True)
        prediction = model.predict(input_variable)
        st.write("Predicted value is :", prediction)
    else:
        st.write('Pls press the predict button for prediction')

with interpret:
    st.subheader('Model Interpretation')
    #st.write(f"CHURN = {model.intercept_.round(2)} + {model.coef_[0].round(2)} TENURE + {model.coef_[1].round(2)} REGULARITY")

    #st.markdown("<br>", unsafe_allow_html= True)

    #st.markdown(f"- The expected Profit for a startup is {model.intercept_}")

    #st.markdown(f"- For every additional 1 dollar spent on R&D Spend, the expected profit is expected to increase by ${model.coef_[0].round(2)}  ")

    #st.markdown(f"- For every additional 1 dollar spent on Administration Expense, the expected profit is expected to decrease by ${model.coef_[1].round(2)}  ")
