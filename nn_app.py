import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Streamlit app title
st.title("Stock Price Prediction Using Neural Networks")

# Fetch default stock data (Apple Inc.) from Yahoo Finance
def fetch_default_data():
    stock_data = yf.download("AAPL", start="2023-01-01", end="2023-12-31")
    stock_data.reset_index(inplace=True)
    stock_data = stock_data[["Open", "High", "Low", "Close", "Volume"]]
    return stock_data

default_df = fetch_default_data()

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

# Select features and target
st.write("### Select Features and Target")
features = st.multiselect("Select Features", data.columns.tolist(), default=data.columns[:-1])
target = st.selectbox("Select Target", data.columns.tolist(), index=len(data.columns) - 1)

if features and target:
    # Prepare data
    X = data[features].values
    y = data[target].values

    # Normalize data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build neural network model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train model
    st.write("### Training the Model")
    epochs = st.slider("Select Number of Epochs", min_value=10, max_value=200, value=50, step=10)
    if st.button("Train Model"):
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, verbose=0)
        st.write("Training Complete")

        # Display training history
        st.write("### Training History")
        history_df = pd.DataFrame(history.history)
        st.line_chart(history_df[['loss', 'val_loss']])

    # Make predictions
    st.write("### Make Predictions")
    new_data = st.text_area("Enter new data for prediction (comma-separated values):")
    if new_data:
        try:
            new_data = np.array([float(x) for x in new_data.split(",")]).reshape(1, -1)
            new_data_scaled = scaler.transform(new_data)
            prediction = model.predict(new_data_scaled)
            st.write(f"Predicted Value: {prediction[0][0]:.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")
