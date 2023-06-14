import streamlit as st
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import re

st.write("""
# Sports Car Price Prediction Application
This application predicts the **Sports Car** price!
""")
####### Import Data #######
current_directory = os.getcwd()
file_name = "Sport car price.csv"
file_path = os.path.join(current_directory, file_name)
df=pd.read_csv(file_path)

####### Data Preprocessing #######
# Converts data to numeric data, those that could not be parsed to numeric are replaced with null value
df['EngineSize'] = pd.to_numeric(df['EngineSize'], errors='coerce')

#Remove values that contain "+", "-", "," symbol
symbols = ["+", "-", ","]
for symbol in symbols:
#     df["EngineSize"] = df["EngineSize"].str.replace(re.escape(symbol), "", regex=True)
    df["Horsepower"] = df["Horsepower"].str.replace(re.escape(symbol), "", regex=True)
    df["Torque"] = df["Torque"].str.replace(re.escape(symbol), "", regex=True)

#Replace <1.9 in MPH column with 1.9
df["MPH"] = df["MPH"].replace("< 1.9", "1.9")

#Convert categorical data to numerical
CarMake_column = df.iloc[:, 0].values.reshape(-1, 1)
onehotencoder = OneHotEncoder()
CarMake_encoded = onehotencoder.fit_transform(CarMake_column).toarray()
categories = onehotencoder.categories_[0]
new_columns = [f'CarMake_{category}' for category in categories]
df_encoded = pd.concat([df.drop(columns='CarMake'), pd.DataFrame
                        (CarMake_encoded, columns=new_columns)], axis=1)
df['CarMake'] = df_encoded[new_columns].values.argmax(axis=1)

# Drop rows with missing values
df.dropna(inplace=True)

# Drop rows with empty string 
df = df[df != ""].dropna(how="all", axis=0)
#reset index
df.reset_index(drop=True, inplace=True)

#convert Horsepower, Torque and MPH from string to float type 
df["EngineSize"] = df["EngineSize"].astype("float")
df["Horsepower"] = df["Horsepower"].astype("float")
df["Torque"] = df["Torque"].astype("float")
df["MPH"] = df["MPH"].astype("float")

tab1, tab2, tab3 = st.tabs(["Exploratory Data Analysis", "Prediction", "Performance Evaluation"])

with tab1:
   #Display Dataset info
    st.subheader('Dataset')
    st.write(df.head(10))
    st.markdown("<p style='font-family: Verdana;'>Statistical Analysis</p>", unsafe_allow_html=True)
    st.write(df.describe())

    st.markdown("<p style='font-family: Verdana;'>Boxplot of Engine Size</p>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df["EngineSize"], orient="h", ax=ax)
    ax.set_title("Boxplot of Engine Size")
    # plt.ylim(plt.ylim()[::-1])
    st.pyplot(fig)

    st.markdown("<p style='font-family: Verdana;'>Histogram of MPH</p>", unsafe_allow_html=True)
    data = df["MPH"]
    plt.hist(data, bins=5, edgecolor='black')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of MPH')
    plt.ylim(plt.ylim()[::-1])
    st.pyplot()


with tab2:
    st.subheader('Prediction')
    def user_input_features():
        car_make = st.sidebar.slider('Car Make', 0.00, 37.00, 2.0)
        year = st.sidebar.slider('Year', 1965, 2024, 2022)
        engine_size = st.sidebar.slider('Engine Size', 0.0, 8.4, 1.3)
        horsepower = st.sidebar.slider('Horsepower', 150, 10000, 200)
        torque = st.sidebar.slider('Torque', 151, 7500, 500)
        mph = st.sidebar.slider('MPH', 2.0, 7.0, 5.0)
        data = {'car_make': car_make,
                'year': year,
                'engine_size': engine_size,
                'horsepower': horsepower,
                'torque': torque,
                'mph': mph}
        features = pd.DataFrame(data, index=[0])
        return features

    user_data = user_input_features()

    st.markdown("<p style='font-family: Verdana;'>Move the slider on sidebar to select input values</p>", unsafe_allow_html=True)

    # Create a radio button
    # model = st.radio("Select a regression model", ["Multiple Linear Regression", "Polynomial Regression"])

    ####### Train Machine Learning Model and Prediction #######
    # if model == "Multiple Linear Regression":
    #multiple linear regression
    x=df.iloc[:,:-1].values
    y=df.iloc[:,6].values

    x_train, x_test, y_train, y_test_multi = train_test_split(x, y, test_size = 0.25, random_state = 0)
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.fit_transform(x_test)

    regressor=LinearRegression()
    regressor.fit(x_train,y_train)

    y_predict = regressor.predict(x_test)
    y_predict_multi = [max(value, 0) for value in y_predict]
    y_output_multi = np.round(regressor.predict(user_data), 2)

    #Feature Selection - L2 Regularized
    # Apply Ridge regression with regularization parameter alpha
    alpha = 0.1
    ridge = Ridge(alpha=alpha)
    ridge.fit(x_train, y_train)

    # Predict on the training and testing data
    # y_train_pred_ridged = ridge.predict(x_train)
    y_predict_multi_ridged = ridge.predict(x_test)
    y_output_ridged_multi = np.round(ridge.predict(user_data), 2)
    # st.write("Multiple Linear Regression Prediction", y_output_multi[0])
    # st.write("Multiple Linear Regression Prediction (after L2 Regularization)", y_output_ridged_multi[0])


    # if model == "Polynomial Regression":
    #polynomial regression
    x=df.iloc[:,:-1].values
    y=df.iloc[:,6].values

    x_train, x_test, y_train, y_test_poly = train_test_split(x, y, test_size = 0.25, random_state = 0)
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.fit_transform(x_test)

    poly_reg = PolynomialFeatures(degree=3)
    x_train_poly=poly_reg.fit_transform(x_train)
    x_test_poly=poly_reg.fit_transform(x_test)
    regressor=LinearRegression()
    regressor.fit(x_train_poly,y_train)

    y_predict = regressor.predict(x_test_poly)
    y_predict_poly = [max(value, 0) for value in y_predict]
    y_output = np.round(regressor.predict(poly_reg.fit_transform(user_data)), 2)
    y_output_poly = [max(value, 0) for value in y_output]

    #Feature Selection - L2 Regularized
    # Apply Ridge regression with regularization parameter alpha
    alpha = 0.1
    ridge = Ridge(alpha=alpha)
    ridge.fit(x_train_poly, y_train)

    # Predict on the training and testing data
    # y_train_pred_ridged = ridge.predict(x_train_poly)
    y_predict_ridged = ridge.predict(x_test_poly)
    y_predict_poly_ridged = [max(value, 0) for value in y_predict_ridged]
    y_output_ridged = np.round(ridge.predict(poly_reg.fit_transform(user_data)), 2)
    y_output_ridged_poly = [max(value, 0) for value in y_output_ridged]
    # st.write("Polynomial Regression Prediction", y_output[0])
    # st.write("Polynomial Regression Prediction (after L2 Regularization)", y_output_ridged[0])

    data = {'Column 1': ["Multiple Linear Regression", "Polynomial Regression"], 'Column 2': [y_output_multi, y_output_poly], 'Column 3': [ y_output_ridged_multi, y_output_ridged_poly]}
    prediction_df = pd.DataFrame(data)

    # Set the custom headers
    custom_headers = ["", 'Non-regularized', 'Regularized']
    prediction_df.columns = custom_headers
    df_reset = prediction_df.reset_index(drop=True)
    st.table(prediction_df)


with tab3:
    st.subheader('Performance Evaluation')
    # Performance Evaluation for Multiple Linear Regression
    st.markdown("<p style='font-family: Verdana;'>Multiple Linear Regression</p>", unsafe_allow_html=True)
    # Non-Regularized
    # Mean Squared Error (MSE)
    mse_multi = metrics.mean_squared_error(y_test_multi, y_predict_multi)
    # Root Mean Squared Error (RMSE)
    rmse_multi = metrics.mean_squared_error(y_test_multi, y_predict_multi, squared=False)
    # Mean Absolute Error (MAE)
    mae_multi = metrics.mean_absolute_error(y_test_multi, y_predict_multi)
    # R-squared (R2) Score
    r2_multi = metrics.r2_score(y_test_multi, y_predict_multi)

    # Regularized
    # Mean Squared Error (MSE)
    mse_multi_ridged = metrics.mean_squared_error(y_test_multi, y_predict_multi_ridged)
    # Root Mean Squared Error (RMSE)
    rmse_multi_ridged = metrics.mean_squared_error(y_test_multi, y_predict_multi_ridged, squared=False)
    # Mean Absolute Error (MAE)
    mae_multi_ridged = metrics.mean_absolute_error(y_test_multi, y_predict_multi_ridged)
    # R-squared (R2) Score
    r2_multi_ridged = metrics.r2_score(y_test_multi, y_predict_multi_ridged)

    data = {'Column 1': ["Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "Mean Absolute Error (MAE)", "R-squared (R2) Score"], 'Column 2': [mse_multi, rmse_multi, mae_multi, r2_multi], 'Column 3': [mse_multi_ridged, rmse_multi_ridged, mae_multi_ridged, r2_multi_ridged]}
    prediction_df = pd.DataFrame(data)
    custom_headers = ["", 'Non-regularized', 'Regularized']
    prediction_df.columns = custom_headers
    st.table(prediction_df)

    #Performance Evaluation for Polynomial Regression
    st.markdown("<p style='font-family: Verdana;'>Polynomial Regression</p>", unsafe_allow_html=True)
    # Non-Regularized
    # Mean Squared Error (MSE)
    mse_poly = metrics.mean_squared_error(y_test_poly, y_predict_poly)
    # Root Mean Squared Error (RMSE)
    rmse_poly = metrics.mean_squared_error(y_test_poly, y_predict_poly, squared=False)
    # Mean Absolute Error (MAE)
    mae_poly = metrics.mean_absolute_error(y_test_poly, y_predict_poly)
    # R-squared (R2) Score
    r2_poly = metrics.r2_score(y_test_poly, y_predict_poly)

    # Regularized
    # Mean Squared Error (MSE)
    mse_poly_ridged = metrics.mean_squared_error(y_test_poly, y_predict_poly_ridged)
    # Root Mean Squared Error (RMSE)
    rmse_poly_ridged = metrics.mean_squared_error(y_test_poly, y_predict_poly_ridged, squared=False)
    # Mean Absolute Error (MAE)
    mae_poly_ridged = metrics.mean_absolute_error(y_test_poly, y_predict_poly_ridged)
    # R-squared (R2) Score
    r2_poly_ridged = metrics.r2_score(y_test_poly, y_predict_poly_ridged)

    data = {'Column 1': ["Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "Mean Absolute Error (MAE)", "R-squared (R2) Score"], 'Column 2': [mse_poly, rmse_poly, mae_poly, r2_poly], 'Column 3': [mse_poly_ridged, rmse_poly_ridged, mae_poly_ridged, r2_poly_ridged]}
    prediction_df = pd.DataFrame(data)
    custom_headers = ["", 'Non-regularized', 'Regularized']
    prediction_df.columns = custom_headers
    st.table(prediction_df)











