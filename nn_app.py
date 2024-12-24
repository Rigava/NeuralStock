import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Streamlit APP Config
st.set_page_config(page_title="ML prediction", page_icon=":bar_chart:", layout="wide")
st.title("Stock Price Prediction Using Neural Networks")
with st.expander("Quick Guide"):
    st.write("""Streamlit app now demonstrates a neural network for analyzing past stock prices, 
    trading volume, and other financial indicators. Upload a CSV file with the relevant data to 
    train the model and make predictions""")
# Fetch default stock data (Apple Inc.) from Yahoo Finance
def fetch_default_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, group_by = 'Ticker',start=start_date, end=end_date)
    # Transform the DataFrame: stack the ticker symbols to create a multi-index (Date, Ticker), then reset the 'Ticker' level to turn it into a column
    stock_data = stock_data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    stock_data.index.name = "Date" 
    stock_data.reset_index(inplace=True)
    # stock_data = stock_data[["Date","Open", "High", "Low", "Close", "Volume"]]
    return stock_data

# Sidebar for symbol and date selection
st.sidebar.write("### Select Stock Symbol and Date Range")
stock_symbol = st.sidebar.selectbox("Select Stock Symbol", ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA","NSEI"], index=0)
default_start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
default_end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

# Ensure valid date range
if default_start_date >= default_end_date:
    st.sidebar.error("End Date must be after Start Date.")
    default_df = pd.DataFrame()
else:
    default_df = fetch_default_data(stock_symbol, default_start_date, default_end_date)

# File upload for dataset
uploaded_file = st.file_uploader("Upload a CSV file with stock data", type="csv")

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())
else:
    st.info("No file uploaded. Using default Apple Inc. data from Yahoo Finance.")
    data = default_df
    st.write("### Default Dataset Preview")
    st.dataframe(data)
st.write(f"The shape of actual dataset is {data.shape} and column names are {data.columns.tolist()}")
#Visualisation of the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.Date, y=data['Close'], name='Close', line=dict(color='blue')))
# fig.add_trace(go.Scatter(x=df.Date, y=df['SMA10'], name='fast', line=dict(color='red')))
# fig.add_trace(go.Scatter(x=df.Date, y=df['SMA50'], name='slow', line=dict(color='white')))
fig.update_xaxes(type='category')
fig.update_layout(height=800)
st.plotly_chart(fig,use_container_width=True)
# ------------------------ML model step 1 - Split data into Training and Testing
split = 0.8
len_train = int(len(data)*split)
data_train = data[:len_train]
data_test = data[len_train:]
st.write(f"The shape of training data is set for {data_train.shape} & for test data is {data_test.shape}")
# Select features and target
st.write("### Select Features and Target")
features = st.selectbox("Select one Feature for predicting the close value", data.columns.tolist())
target = st.selectbox("Select Target", data.columns.tolist(), index=len(data.columns) - 1)

if features and target:
    # Normalize feature data    
    X = data_train[features].values
    y = data_train[target].values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(X.reshape(-1,1))
    st.write(f"The shape of scaled data of the closing price in training dataset is {scaled_data.shape}")
    # Prepare data
    prediction_days = 60
    x_train =[]
    y_train =[]
    for x in range(prediction_days , len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days : x, 0]) 
        y_train.append(scaled_data[x, 0])
    x_train , y_train = np.array(x_train), np.array(y_train)
    st.write(x_train)
    # Adding an extra 3rd dimension by Reshaping the array so that it work with NN model
    x_train =np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    st.write(f"The feature engineering dimension is set for {x_train.shape}")
    epochs = st.slider("Select Number of Epochs", min_value=10, max_value=200, value=50, step=10)
#   # ---------------Train model
    if st.button("Train Model"):
        st.spinner("### Training the Model")
    

        # # Build neural network model
        model = Sequential()
        model.add(LSTM(units = 50, return_sequences=True,input_shape=(x_train.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 50))
        model.add(Dropout(0.2))
        model.add(Dense(units= 1))

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history = model.fit(x_train, y_train, epochs = epochs, batch_size = 32)
        st.write("Training Complete")
  

        # ---------------Preparing the Test Dataset
        total_dataset = pd.concat((data_train['Close'],data_test['Close']),axis=0)
        # dataset for what the model can see through---- model input
        model_inputs = total_dataset[len(total_dataset)-len(data_test)-prediction_days :].values
        
        scaled_data_test = scaler.transform(model_inputs.reshape(-1,1)) # Choosing Close values from the training dataset only for prediction
        st.write(f"The model input dimension is set for {scaled_data_test.shape}")
        x_test = []
        for x in range(prediction_days , len(scaled_data_test)):
            x_test.append(scaled_data_test[x-prediction_days : x, 0]) 
        x_test = np.array(x_test)  
        st.write(x_test)  
        # Adding an extra 3rd dimension by Reshaping the array so that it work with NN model
        x_test =np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        st.write(f"The feature engineering dimension for test data is set for {x_test.shape}") 
        


#     # Make predictions & visualize prediction
        st.write("### Make Predictions")
        predict_prices = model.predict(x_test)
        predict_prices = scaler.inverse_transform(predict_prices)

        plt.plot(data_test['Close'].values, color = "blue", label="Actual price")
        plt.plot(predict_prices, color = "black", label="Predcited price")
        plt.title("NN model for stock prediction")
        plt.xlabel('Time')
        plt.legend()
        st.pyplot(plt)
        #Predict next day
        # model_inputs = model_inputs.reshape(-1,1)
        model_inputs = scaler.transform(model_inputs.reshape(-1,1))
        real_data = [model_inputs[len(model_inputs)+1-prediction_days : len(model_inputs+1), 0]]
        real_data = np.array(real_data)
        st.write(real_data)
        real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))
        prediction = model.predict(real_data)
        st.write(f"the predicted value for the next day is {scaler.inverse_transform(prediction)}")
                





#     new_data = st.text_area("Enter new data for prediction (comma-separated values):")
#     if new_data:
#         try:
#             new_data = np.array([float(x) for x in new_data.split(",")]).reshape(1, -1)
#             new_data_scaled = scaler.transform(new_data)
#             prediction = model.predict(new_data_scaled)
#             st.write(f"Predicted Value: {prediction[0][0]:.2f}")
#         except Exception as e:
#             st.error(f"Error in prediction: {e}")
