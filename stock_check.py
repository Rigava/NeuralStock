import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Fetch stock data using yfinance
def fetch_default_data(symbol, start_date, end_date):
    try:
        stock_data = yf.download(symbol, group_by='Ticker', start=start_date, end=end_date)
        # Transform the DataFrame: stack the ticker symbols to create a multi-index (Date, Ticker), then reset the 'Ticker' level to turn it into a column
        stock_data = stock_data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
        stock_data.index.name = "Date" 
        stock_data.reset_index(inplace=True)
        return stock_data
    except Exception as e:
        st.error(f"Could not fetch data for {symbol} from Yahoo Finance. {e}")
        return None

# Load and filter data from the uploaded CSV file
def process_uploaded_csv(uploaded_file, stocks, start_date, end_date):
    try:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file, parse_dates=['Date'])
        st.write("Uploaded CSV Data Preview:")
        st.write(data.head())  # Display the first few rows of the uploaded data
        
        # Filter data for the requested stocks
        data = data[data['Ticker'].isin(stocks)]
        st.write("Filtered Data by Stocks:")
        st.write(data.head())  # Display the filtered data
        
        # Filter data for the specified date range
        data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))]
        st.write("Filtered Data by Date Range:")
        st.write(data.head())  # Display the filtered data
        
        if data.empty:
            st.warning("No matching data found in the uploaded CSV file for the given stocks or date range.")
        return data
    except Exception as e:
        st.error(f"Error processing the uploaded CSV file: {e}")
        return None

# Streamlit App
def main():
    st.title("Stock Cumulative Returns Visualization")
    st.sidebar.header("Configuration")
    
    # Input fields for stock symbols and date range
    stocks = st.sidebar.text_input("Enter stock symbols (comma-separated):", "AAPL, GOOGL, MSFT").split(",")
    stocks = [stock.strip().upper() for stock in stocks]
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-01"))
    
    # File uploader for the default CSV
    uploaded_file = st.sidebar.file_uploader("Upload Default CSV File", type=["csv"])
    
    if st.sidebar.button("Fetch and Plot Data"):
        st.write("### Stock Cumulative Returns")
        cumulative_returns = pd.DataFrame()
        
        # Check if an uploaded file is provided
        csv_data = None
        if uploaded_file:
            st.info("Processing uploaded CSV file...")
            csv_data = process_uploaded_csv(uploaded_file, stocks, start_date, end_date)
        
        for stock in stocks:
            stock_data = None
            
            # If CSV data is available, filter it for the specific stock
            if csv_data is not None:
                stock_data = csv_data[csv_data['Ticker'] == stock]
            
            # Otherwise, fetch data from Yahoo Finance
            if stock_data is None or stock_data.empty:
                st.warning(f"Falling back to Yahoo Finance for {stock}.")
                stock_data = fetch_default_data(stock, start_date, end_date)
            
            if stock_data is None or stock_data.empty or 'Close' not in stock_data.columns:
                st.error(f"Could not retrieve data for {stock}. Skipping.")
                continue
            
            # Calculate daily returns and cumulative returns
            stock_data.set_index('Date', inplace=True)
            daily_return = stock_data['Close'].pct_change()
            cumulative_returns[stock] = (1 + daily_return).cumprod() - 1
        
        # Plot the cumulative returns
        if not cumulative_returns.empty:
            plt.figure(figsize=(12, 6))
            for stock in cumulative_returns.columns:
                plt.plot(cumulative_returns.index, cumulative_returns[stock], label=stock)
            
            plt.title('Cumulative Returns of Multiple Stocks', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Cumulative Return', fontsize=12)
            plt.legend(title="Stocks")
            plt.grid(alpha=0.3)
            
            # Display the plot in Streamlit
            st.pyplot(plt)
        else:
            st.warning("No data to plot. Please check the stock symbols, date range, or uploaded file.")

if __name__ == "__main__":
    main()
