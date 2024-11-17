# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')


# Set up the app title and image
st.title('Traffic Volume Predictor 🚗')
# st.image('traffic_image.gif', use_column_width = True)
         
# NOTE: In Streamlit, use_column_width=True within st.image automatically 
# adjusts the image width to match the width of the column or container in 
# which the image is displayed. This makes the image responsive to different 
# screen sizes and layouts, ensuring it scales nicely without needing to 
# specify exact pixel dimensions.

st.write("Utilize our advanced Machine Learning application to predict traffic volume.") 
# st.image('traffic_image.gif', use_column_width = True)

# Reading the pickle file that we created before 
model_pickle = open('traffic.pickle', 'rb') 
reg_model = pickle.load(model_pickle) 
model_pickle.close()


# Sidebar for user inputs with expanders
with st.sidebar:
    st.image('traffic_sidebar.jpg', use_column_width=True)
    st.title('Traffic Volume Predictor')
    with st.expander("Option 1: Upload Your CSV File"):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload a CSV file with traffic details to pre-fill the form.")
        # Display Example CSV
        st.header('Sample Data Format for Upload')

        # Import example dataframe
        df = pd.read_csv('Traffic_Volume.csv')

        # Convert the 'date_time' column to datetime format
        df['date_time'] = pd.to_datetime(df['date_time'], format='%m/%d/%y %H:%M')

        # Extract specific month, day, and hour, and add to the dataframe as new columns
        df['month'] = df['date_time'].dt.month_name()  # Month name
        df['weekday'] = df['date_time'].dt.day_name()  # Day name
        df['hour'] = df['date_time'].dt.hour           # Hour (0-23)

        # Format the dataframe correctly by dropping unnecessary columns
        df_clean = df.drop(columns=['date_time', 'traffic_volume'])

        # Write the cleaned dataframe to Streamlit
        st.write(df_clean.head())

        # Warning message to users
        st.warning('Ensure your uploaded file has the same column names and data types as shown above.')

    # Manual entry option in the sidebar
    with st.sidebar.expander("Option 2: Fill Out Form"):
        with st.form('manual_entry_form'):
            st.write('Enter the traffic details manually using the form below.')
            holiday = st.selectbox("Holiday", options=['No holiday', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Christmas Day', 'New Year\'s Day', 'Washington\'s Birthday', 'Memorial Day', 'Independence Day', 'State Fair', 'Labor Day', 'Martin Luther King Jr Day'])
            temperature_kelvin = st.number_input("Average temperature in Kelvin", min_value=243.00, max_value=311.00, step=0.01, value=273.15)
            rainfall_mm = st.number_input("Rainfall (mm)", min_value=0.00, max_value=9831.3, step=0.01, value=0.00)
            snowfall_mm = st.number_input("Snowfall (mm)", min_value=0.00, max_value=0.55, step=0.01, value=0.00)
            cloud_coverage = st.number_input("Cloud Coverage (%)", min_value=0, max_value=100, step=1, value=0)
            weather = st.selectbox("Weather Condition", options=['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 'Rain', 'Smoke', 'Snow', 'Squall', 'Thunderstorm'])
            month = st.selectbox("Month", options=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
            day_of_week = st.selectbox("Day of the Week", options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            hour = st.selectbox("Hour of the Day", options=list(range(0, 24)))                       
            submit_button = st.form_submit_button("Predict")

# Recall original data frame for dummy creation

# Check to see if CSV file has been uploaded

if uploaded_file is not None:
    # Success message
    st.success('CSV file uploaded successfully')
    user_df = pd.read_csv(uploaded_file)

    # Create slider for alpha value selection
    alpha_input = st.slider('Select alpha value for prediction intervals', min_value=0.01, max_value=0.50, step=0.01)

        # Create Necessary Dummies
    # Input features
    #features = user_df[['holiday', 'temperature_kelvin', 'rainfall_mm', 'snowfall_mm', 'cloud_coverage', 'weather', 'month', 'day_of_week', 'hour']]
    features = user_df[['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'month', 'weekday', 'hour']]
    
    # Combine input data into original data frame
    # Append features to df_orig
    combined_df = pd.concat([df.drop(columns=['date_time', 'traffic_volume']), features], ignore_index=True)
    #Chance column name to match what model expects
    combined_df.rename(columns={'weekday': 'day_of_week'}, inplace=True)
    # Fill empty holiday with "no holiday"
    combined_df['holiday'] = combined_df['holiday'].fillna('No holiday')
    
    # One-hot encoding to handle categorical variables
    combined_features_encoded = pd.get_dummies(combined_df)

    # Pull out user CSV from combined dataframe
    features_encoded = combined_features_encoded.iloc[len(df):].reset_index(drop=True)
    
    # Run predictions for each row of data
    predictions, intervals = reg_model.predict(features_encoded, alpha=alpha_input)

    # Storing results in the original dataframe
    user_df["predicted Traffic Volume"] = predictions.round(0)
    user_df["Lower Traffic Volume Limit"] = np.maximum(intervals[:, 0], 0).round(0)
    user_df["Upper Traffic Volume Limit"] = intervals[:, 1].round(0)

    # Display results
    st.header(f"Here is your Traffic Volume Prediction Results with {(1 - alpha_input) * 100:.2f}% Confidence Interval")
    st.write(user_df)

    # If the form data is submitted successfully
else:
    # if submit_button:
        alpha_input = st.slider('Select alpha value for prediction intervals', min_value=0.01, max_value=0.50, step=0.01)

        # Encode the form inputs for model prediction
        df_clean = df.drop(columns=['date_time', 'traffic_volume'])

        encode_df = df_clean.copy()

        encode_df.loc[len(encode_df)] = [holiday, temperature_kelvin, rainfall_mm, snowfall_mm, cloud_coverage, weather, month, day_of_week, hour]

               # Ensure the data types match the original data's types
        encode_df = encode_df.astype(df_clean.dtypes)

        encode_df.rename(columns={'weekday': 'day_of_week'}, inplace=True)
        # Fill empty holiday with "no holiday"
        encode_df['holiday'] = encode_df['holiday'].fillna('No holiday')

        # Get Dummies for categorical variables
        df_dummy_form = pd.get_dummies(encode_df)

        # Extract the encoded user data (last row)
        user_form_encoded_df = df_dummy_form.tail(1)

        # Run prediction for the user data
        prediction, intervals = reg_model.predict(user_form_encoded_df, alpha=alpha_input)

        # Extract prediction and interval values
        pred_value = prediction[0]
        lower_limit = intervals[:, 0][0]
        upper_limit = intervals[:, 1][0]

        # Ensure limits are within [0, 1]
        lower_limit = max(0, lower_limit)

        # Display the prediction results
        # st.header("Predicted Traffic Volume...")
        # st.metric('Predicted Traffic Volume', f"{pred_value:.2f}")
        # st.write(f"CONFIDENCE INTERVAL ({(1 - alpha_input) * 100:.2f}%): [Lower limit: {lower_limit:.2f}, Upper limit: {upper_limit:.2f}]")

        # Display the prediction results
       # Ensure variables are scalar values
        pred_value = pred_value.item() if isinstance(pred_value, np.ndarray) else pred_value
        lower_limit = lower_limit.item() if isinstance(lower_limit, np.ndarray) else lower_limit
        upper_limit = upper_limit.item() if isinstance(upper_limit, np.ndarray) else upper_limit

        # Display the prediction results
        st.header("Predicted Traffic Volume")
        st.metric(label="Predicted Traffic Volume", value=f"{pred_value:.2f}")
        confidence_percentage = (1 - alpha_input) * 100
        st.write(
            f"Confidence Interval ({confidence_percentage:.2f}%): "
            f"[Lower Limit: {lower_limit:.2f}, Upper Limit: {upper_limit:.2f}]"
        )

#The extract prediction and interval values was not working... chat gpt fixed it above

# Model Insights
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                                  "Histogram of Residuals", 
                                  "Predicted Vs. Actual", 
                                  "Coverage Plot"])

with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")

with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")

with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")

with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")
