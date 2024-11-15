import streamlit as st
import yfinance as yf
import mplfinance as mpf
from plotly import graph_objects as go
import ta  # Import the ta library
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
st.title("Stock Prediction and Portfolio Optimization App")
tab1,tab2,tab3=st.tabs(['Stock Info','Forecaste Using Ml Models','Monte Carlo Simulation'])
# Title and introduction

with tab1:

    # Dropdown for stock tickers
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "INTC", "AMD",
        "SPY", "QQQ", "V", "PYPL", "CSCO", "CRM", "WMT", "DIS", "NKE", "JNJ",
        "PFE", "BABA", "IBM", "KO", "GS", "AXP", "BA", "GE", "CAT", "RTX",
        "HD", "MCD", "PEP", "UNH", "MRK", "CVX", "XOM", "GS", "USB", "WFC",
        "C", "BAC", "UNP", "MMM", "T", "VZ", "INTU", "ADBE", "ORCL", "QCOM",
        "MU", "XLNX", "LMT", "UPS", "F", "GM", "TGT", "LOW", "EXXON", "FIS",
        "ISRG", "MDT", "ABT", "SYK", "BMY", "GILD", "AMGN", "BIIB", "LLY", "MO",
        "PM", "KO", "PEP", "CL", "PG", "UN", "COST", "SBUX", "TSCO", "DE", "ADM",
        "CHD", "STZ", "NEM", "HUM", "CI", "ANTM", "AIG", "TMO", "RMD", "SIVB",
        "EL", "MELI", "FISV", "DXCM", "REGN", "VRTX", "TSM", "MKC", "SYY", "LULU",
        "ROST", "SHW", "CHTR", "ZTS", "TROW", "BKNG", "CVS", "MCO", "NEE", "AES",
        "XEL", "ED", "PPL", "PEG", "DTE", "FRT", "O", "SPG", "PLD", "AMT", "CCI",
        "EQIX", "PSA", "AVB", "EQR", "IRM", "EXR", "MAA", "ESS", "UMH", "APTS"
    ]

    ticker = st.selectbox("Select a stock ticker", tickers)

    # Date range input
    start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))

    # Download stock data for the selected ticker
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Retrieve additional info for the selected ticker
    stock_info = yf.Ticker(ticker).info

    # Display stock data and metrics
    st.subheader(f"{stock_info['shortName']} ({ticker}) Stock Data")
    st.write(stock_data)

    # Display additional metrics
    st.write(f"Max Close Price: {stock_data['Close'].max()}")
    st.write(f"Min Close Price: {stock_data['Close'].min()}")
    st.write("Summary Statistics:")
    st.write(stock_data.describe())

    # Organize the company information into a DataFrame for tabular display
    company_info = {
        "Company Name": stock_info.get('shortName', 'N/A'),
        "Average Volume": stock_info.get('averageVolume', 'N/A'),
        "Market Cap": stock_info.get('marketCap', 'N/A'),
        "Previous Close": stock_info.get('previousClose', 'N/A'),
        "52-Week High": stock_info.get('fiftyTwoWeekHigh', 'N/A'),
        "52-Week Low": stock_info.get('fiftyTwoWeekLow', 'N/A'),
        "200-Day Average": stock_info.get('twoHundredDayAverage', 'N/A'),
        "Short Ratio": stock_info.get('shortRatio', 'N/A')
    }

    company_info_df = pd.DataFrame(list(company_info.items()), columns=["Metric", "Value"])

    # Display the company info in a tabular format
    st.subheader(f"{ticker} Company Information")
    st.dataframe(company_info_df)

    # Render the clickable website link separately
    website_url = stock_info.get('website', '#')
    if website_url != '#':
        st.markdown(f"**Website**: [Visit {stock_info.get('shortName')} Website]({website_url})")
    else:
        st.markdown("**Website**: N/A")

    # Plot closing price with Moving Averages
    st.subheader(f"{ticker} Closing Price & Moving Averages")
    stock_data['20_MA'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['50_MA'] = stock_data['Close'].rolling(window=50).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(stock_data['Close'], label='Close Price', color='blue')
    plt.plot(stock_data['20_MA'], label='20-Day MA', color='orange')
    plt.plot(stock_data['50_MA'], label='50-Day MA', color='green')
    plt.title(f"{ticker} Closing Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    st.pyplot(plt)



with tab3:
    st.title("Monte Carlo Simulation for Portfolio Optimization")

    # User selects the stocks
    selected_stocks = st.multiselect("Select Stocks", tickers, default=['GOOGL', 'AAPL'])

    # Ensure that the user selects at least 2 stocks
    if len(selected_stocks) < 2:
        st.warning("Please select at least 2 stocks.")
    else:
        # Download stock data
        data = yf.download(selected_stocks, start="2020-01-01", end="2024-01-01")['Adj Close']
        st.write(f"Data for selected stocks: {selected_stocks}")

        # Calculate daily returns
        returns = data.pct_change().dropna()

        # Monte Carlo Simulation Parameters
        num_simulations = 10000
        num_assets = len(selected_stocks)
        risk_free_rate = 0.01  # Risk-free rate (1% for example)

        # Arrays to store simulation results
        results = np.zeros((3, num_simulations))

        for i in range(num_simulations):
            # Generate random portfolio weights that sum to 1
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)  # Normalize to sum to 1

            # Portfolio return and volatility (standard deviation)
            portfolio_return = np.sum(weights * returns.mean()) * 252  # Annualize return
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualized volatility

            # Calculate Sharpe ratio (using risk-free rate)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

            # Store results
            results[0, i] = portfolio_return
            results[1, i] = portfolio_volatility
            results[2, i] = sharpe_ratio

        # Convert results into a DataFrame for better readability
        simulation_df = pd.DataFrame(results.T, columns=["Return", "Volatility", "Sharpe Ratio"])

        # Find the portfolio with the maximum Sharpe ratio
        max_sharpe_idx = simulation_df['Sharpe Ratio'].idxmax()
        max_sharpe_portfolio = simulation_df.iloc[max_sharpe_idx]
        best_weights = np.random.random(num_assets)
        best_weights /= np.sum(best_weights)  # Best weights for the optimal portfolio

        # Format the portfolio allocation into a DataFrame for display
        portfolio_df = pd.DataFrame({
            'Stock': selected_stocks,
            'Allocation': best_weights.round(3)  # Round the allocation to 3 decimal places
        })

        # Display the best portfolio allocation
        st.write("Best Portfolio Allocation:")
        st.dataframe(portfolio_df)

        # Display the performance metrics for the best portfolio
        st.write(f"Max Sharpe Portfolio Return: {max_sharpe_portfolio['Return']*100:.2f}%")
        st.write(f"Max Sharpe Portfolio Volatility: {max_sharpe_portfolio['Volatility']*100:.2f}%")
        st.write(f"Max Sharpe Portfolio Sharpe Ratio: {max_sharpe_portfolio['Sharpe Ratio']:.2f}")

        # Plot the Monte Carlo Simulations (Risk vs Return)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(simulation_df.Volatility, simulation_df.Return, c=simulation_df['Sharpe Ratio'], cmap='YlGnBu', marker='o')
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Return")
        ax.set_title("Monte Carlo Simulation: Risk vs Return")

        # Highlight the best Sharpe ratio
        ax.scatter(max_sharpe_portfolio['Volatility'], max_sharpe_portfolio['Return'], color='red', marker='*', s=200, label="Max Sharpe Ratio")
        ax.legend(loc='best')

        st.pyplot(fig)

        # Display Pie chart for the best portfolio allocation
        fig_pie = go.Figure(data=[go.Pie(labels=selected_stocks, values=best_weights.round(3)*100)])
        fig_pie.update_layout(title="Best Portfolio Allocation", showlegend=True)
        st.plotly_chart(fig_pie)

with tab2:
    st.title("Forecasting using ML Models")

    # Create a dropdown menu for the user to select the dataset
    selected_ticker = st.selectbox(
        "Select Stock Dataset for Prediction",
        ["AAPL", "GOOGL", "NKE", "IBM", "JNJ", "KO", "MSFT", "NFLX"]
    )

    # Create a slider for selecting the number of days to predict
    days_to_predict = st.slider(
        "Select Number of Days to Predict",
        min_value=1, 
        max_value=365,  # You can adjust this based on your requirements
        value=10,       # Default value (you can change it)
        step=1          # Step size
    )

    # Map selected ticker to its respective LSTM model file path
    model_paths = {
        "AAPL":'lstm_Apple.keras',
        "GOOGL":'lstm_google.keras',
        "NKE": 'lstm_nike.keras',
        "IBM": 'lstm_ibm.keras',
        "JNJ": 'lstm_jnj.keras',
        "KO":'lstm_ko.keras',
        "MSFT":'lstm_msft.keras',
        "NFLX":'lstm_netflix.keras'
    }

    # Load the selected model
    model = load_model(model_paths[selected_ticker])

    # Fetch the selected stock's data (last 1 year)
    data = yf.download(selected_ticker, period='1y', interval='1d')

    # Select 'Adj Close' prices for prediction
    closing_prices = data['Adj Close'].values.reshape(-1, 1)
    
    # Scale the data for prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_prices)

    # Initialize list to store predicted prices
    predicted_prices = []

    # Prepare the most recent 60 days of scaled data as input for the first prediction
    current_batch = scaled_data[-60:].reshape(1, 60, 1)

    # Predict prices for the specified number of future days
    for _ in range(days_to_predict):
        next_prediction = model.predict(current_batch)
        next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
        current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
        predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])

    # Create a list of dates for the future predictions starting from the day after the last date in the data
    last_date = data.index[-1]
    next_day = last_date + pd.Timedelta(days=1)
    prediction_dates = pd.date_range(start=next_day, periods=days_to_predict)

    # Create a DataFrame for the predicted prices with dates as index
    predicted_data = pd.DataFrame(index=prediction_dates, data=predicted_prices, columns=['Adj Close'])

    # Display predicted and historical data side by side
    col1, col2 = st.columns([1, 1])  # Create two columns with equal width
    with col1:
        st.write(f"Predicted Data for {selected_ticker} (Next {days_to_predict} Predictions)")
        st.write(predicted_data.head(days_to_predict))  # Display predicted prices for selected days

    with col2:
        st.write(f"Historical Data for {selected_ticker} (Last {days_to_predict} Historical Prices)")
        st.write(data['Adj Close'].tail(days_to_predict))  # Display last 10 historical prices



    

    # Prepare the candlestick data (with Open, High, Low, Close)
    candlestick_data = data[['Adj Close']].copy()
    candlestick_data['Open'] = candlestick_data['Adj Close']
    candlestick_data['High'] = candlestick_data['Adj Close']
    candlestick_data['Low'] = candlestick_data['Adj Close']
    candlestick_data['Close'] = candlestick_data['Adj Close']
    candlestick_data['Volume'] = 0  # Set volume to 0 as we are not plotting it

    # Creating a candlestick chart and overlaying predictions
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plotting candlesticks
    mpf.plot(candlestick_data, type='candle', style='charles', ax=ax, show_nontrading=True)

    # Overlay the predicted prices as a line plot
    ax.plot(prediction_dates, predicted_prices, linestyle='-', marker='o', color='red', label='Predicted Adj Close')

    # Add title, labels, and ensure proper axis limits
    ax.set_title(f"{selected_ticker} Adjusted Close Price Prediction for Next {days_to_predict} Days")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()

    # Adjust the x-axis to ensure proper display of dates
    ax.set_xticks(prediction_dates)
    ax.set_xticklabels(prediction_dates.strftime('%Y-%m-%d'), rotation=45)

    # Show the plot in Streamlit
    st.pyplot(fig)
