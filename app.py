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
tab4, tab1,tab2,tab3=st.tabs(['About This App', 'Stock Exploration','Price Predictions Using ML Models','Portfolio Allocation Optimization'])

with tab4:
    st.title("📖 App Overview")
    
    # App Introduction
    st.markdown("""  
    Welcome to the **Stock Prediction and Portfolio Optimization App**! This app is a comprehensive tool designed for stock market enthusiasts, investors, and analysts who want to:
    - **Explore stock data**, 
    - **Predict future prices**, and 
    - **Optimize investment portfolios** using advanced data science techniques.
    """)

    # Features Section
    st.subheader("Key Features")
    st.markdown("""  
    1. **Stock Exploration**  
       Dive deep into the performance of your favorite stocks with the following tools:
       - Historical stock price trends
       - Detailed company information
       - Visualizations of key moving averages (20-day and 50-day)
       - Insightful metrics to track stock performance
    
    2. **Price Predictions Using LSTM**  
       Make informed predictions with the power of **Deep Learning**:  
       - **LSTM (Long Short-Term Memory)** models analyze historical stock prices to forecast future trends.  
       - Predict up to **30 days** into the future based on patterns in historical data.  
       - Get a better understanding of potential price movements to aid in decision-making.
    
    3. **Portfolio Allocation Optimization**  
       Optimize your investments with **Monte Carlo Simulations**:  
       - Discover the best allocation of stocks in your portfolio.  
       - Maximize the **Sharpe Ratio** for a balanced trade-off between risk and return.  
       - This feature helps you build a portfolio that aligns with your financial goals and risk tolerance.
    """)

    # About the Author
    st.subheader("👨‍💻 About the Author")
    st.markdown("""  
    Hi, I'm **David Heller**, a data scientist with a robust background in finance and machine learning.  
    - I created this app to combine my expertise in data analysis, financial markets, and advanced modeling techniques.  
    - My goal is to provide an intuitive, data-driven tool to empower investors in making smarter decisions.
    
    Feel free to connect with me or explore more of my work:  
    - [LinkedIn](https://www.linkedin.com/in/david-heller-w/)  
    - [GitHub](https://github.com/davidhellerw)  
    - [Personal Website](https://davidhellerw.com/)
    """)



with tab1:
    st.title("📊 Stock Info")
    
    # Dropdown for stock tickers
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "INTC", "AMD",
        "SPY", "QQQ", "V", "PYPL", "CSCO", "CRM", "WMT", "DIS", "NKE", "JNJ",
        "PFE", "BABA", "IBM", "KO", "GS", "AXP", "BA", "GE", "CAT", "RTX",
        "HD", "MCD", "PEP", "UNH", "MRK", "CVX", "XOM", "GS", "USB", "WFC",
        "C", "BAC", "UNP", "MMM", "T", "VZ", "INTU", "ADBE", "ORCL", "QCOM",
        "MU", "XLNX", "LMT", "UPS", "F", "GM", "TGT", "LOW", "XOM", "FIS",
        "ISRG", "MDT", "ABT", "SYK", "BMY", "GILD", "AMGN", "BIIB", "LLY", "MO",
        "PM", "KO", "PEP", "CL", "PG", "UN", "COST", "SBUX", "TSCO", "DE", "ADM",
        "CHD", "STZ", "NEM", "HUM", "CI", "ANTM", "AIG", "TMO", "RMD", "SIVB",
        "EL", "MELI", "FISV", "DXCM", "REGN", "VRTX", "TSM", "MKC", "SYY", "LULU",
        "ROST", "SHW", "CHTR", "ZTS", "TROW", "BKNG", "CVS", "MCO", "NEE", "AES",
        "XEL", "ED", "PPL", "PEG", "DTE", "FRT", "O", "SPG", "PLD", "AMT", "CCI",
        "EQIX", "PSA", "AVB", "EQR", "IRM", "EXR", "MAA", "ESS", "UMH", "APTS"
    ]

    ticker = st.selectbox("🪙 Select a stock ticker", tickers)

    # Date range input
    start_date = st.date_input("📅 Start Date", value=pd.to_datetime("2023-01-01"))
    end_date = st.date_input("📅 End Date", value=pd.to_datetime("2024-01-01"))

    # Download stock data for the selected ticker
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Retrieve additional info for the selected ticker
    stock_info = yf.Ticker(ticker).info

    # Display stock data and metrics
    st.subheader(f"{stock_info['shortName']} ({ticker}) Stock Data")
    st.write(stock_data)

    # Display additional metrics
    # Extract the max and min close prices and round to 2 decimals
    max_close_price = round(stock_data['Close'].max().item(), 2) 
    min_close_price = round(stock_data['Close'].min().item(), 2)  

    st.write(f"Max Close Price: ${max_close_price}")  # Display as USD
    st.write(f"Min Close Price: ${min_close_price}")  # Display as USD
    
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

    st.subheader("Understanding the Company Metrics")
    st.markdown("""
    - **Company Name**: The official name of the company associated with the selected stock ticker.
    - **Average Volume**: The average number of shares traded daily over a specific period. It gives an idea of the stock's liquidity, where higher values generally indicate more actively traded stocks.
    - **Market Cap (Market Capitalization)**: The total market value of a company’s outstanding shares. It is calculated by multiplying the current share price by the total number of outstanding shares. This metric indicates the company's size and overall value in the market.
    - **Previous Close**: The stock's last closing price from the most recent trading day. It provides a reference point for the stock's performance compared to its current trading price.
    - **52-Week High and Low**: The highest and lowest stock prices recorded over the past 52 weeks. These metrics show the stock's price range and help gauge its volatility.
    - **200-Day Average**: The average stock price over the last 200 trading days. This is often used as a trend indicator, showing whether the stock is trading above or below its long-term average price.
    - **Short Ratio**: The ratio of shares sold short (borrowed and sold in anticipation of a price drop) to the average daily trading volume. A high short ratio may indicate bearish sentiment or a potential short squeeze if the stock's price rises unexpectedly.
    """)

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

    # Explanation for the chart
    st.markdown("""
    ### How to Read the Chart
    - **Closing Price (Blue Line)**: Represents the stock's closing price for each trading day.
    - **20-Day Moving Average (Orange Line)**: A short-term trend indicator.  
      - It calculates the average closing price over the past 20 trading days.  
      - A rising 20-day moving average often indicates a short-term uptrend.
    - **50-Day Moving Average (Green Line)**: A medium-term trend indicator.  
      - It calculates the average closing price over the past 50 trading days.  
      - It's generally used to identify longer-term trends compared to the 20-day moving average.
    
    #### Key Insights:
    1. **Crossovers**:  
       - When the 20-day moving average crosses above the 50-day moving average, it might indicate a bullish signal (buying opportunity).  
       - Conversely, when the 20-day crosses below the 50-day moving average, it might indicate a bearish signal (selling opportunity).
    2. **Trend Confirmation**:  
       - Both moving averages moving upwards can confirm an uptrend.  
       - Both moving downwards can indicate a downtrend.
    3. **Support/Resistance**:  
       - Moving averages can act as dynamic support or resistance levels for the stock price.
    """)


with tab3:
    st.title("📊 Monte Carlo Simulation for Portfolio Optimization")

    # User selects the stocks
    selected_stocks = st.multiselect("🪙 Select Stocks", tickers, default=['GOOGL', 'AAPL'])

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
    
        # Initialize variables to track the best Sharpe Ratio and corresponding weights
        max_sharpe_ratio = -np.inf  # Set to a very low value
        optimal_weights = None  # Placeholder for the weights of the best portfolio
    
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
    
            # Update max Sharpe Ratio and save corresponding weights
            if sharpe_ratio > max_sharpe_ratio:
                max_sharpe_ratio = sharpe_ratio
                optimal_weights = weights  # Save weights for the best portfolio
    
        # Convert results into a DataFrame for better readability
        simulation_df = pd.DataFrame(results.T, columns=["Return", "Volatility", "Sharpe Ratio"])
    
        # Use the optimal weights for the portfolio allocation display
        portfolio_df = pd.DataFrame({
            'Stock': selected_stocks,
            'Allocation': optimal_weights.round(3)  # Use saved optimal weights
        })
    
        # Display the best portfolio allocation
        st.write("Best Portfolio Allocation:")
        st.dataframe(portfolio_df)
    
        # Calculate metrics for the portfolio with the maximum Sharpe Ratio
        best_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(returns.cov() * 252, optimal_weights)))
        best_return = np.sum(optimal_weights * returns.mean()) * 252
    
        # Display the performance metrics for the best portfolio
        st.write(f"Max Sharpe Ratio Portfolio Return: {best_return*100:.2f}%")
        st.write(f"Max Sharpe Ratio Portfolio Volatility: {best_volatility*100:.2f}%")
        st.write(f"Max Sharpe Ratio: {max_sharpe_ratio:.2f}")

        # Add an explanation for these metrics
        st.markdown("""
        **What these values mean:**
        
        - **Max Sharpe Ratio Portfolio Return:** This is the expected annualized return of the portfolio with the maximum Sharpe Ratio, based on historical data. For example, if the return is 20%, you can expect a USD 1,000 investment to grow to approximately USD 1,200 over one year under similar market conditions.
        
        - **Max Sharpe Ratio Portfolio Volatility:** This represents the expected annualized risk (volatility) of the portfolio, meaning the percentage by which returns might fluctuate. 
        
        - **Max Sharpe Ratio:** This measures the portfolio's risk-adjusted return. A Sharpe Ratio of 0.8, for example, means that for every unit of risk, the portfolio generates 0.8 units of excess return above the risk-free rate.

        **Risk-Free Rate Assumption:**
        - The risk-free rate is assumed to be **1%** (0.01) annually in this analysis. This reflects the approximate yield on short-term government bonds, which are considered low-risk investments. Adjusting the risk-free rate to reflect current market conditions may change the Sharpe Ratio but not the portfolio's allocation.
        """)
    
        # Plot the Monte Carlo Simulations (Risk vs Return)
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(simulation_df.Volatility, simulation_df.Return, c=simulation_df['Sharpe Ratio'], cmap='YlGnBu', marker='o')

        # Add colorbar to indicate Sharpe Ratio
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Sharpe Ratio")  # Label the colorbar

        ax.set_xlabel("Expected Annualized Volatility")
        ax.set_ylabel("Expected Annualized Return")
        ax.set_title("Portfolio Optimization: Expected Return vs. Risk")
    
        # Highlight the best Sharpe ratio
        ax.scatter(best_volatility, best_return, color='red', marker='*', s=200, label="Max Sharpe Ratio")
        ax.legend(loc='best')
    
        st.pyplot(fig)

        # Add an explanation for the Risk vs Return Chart
        st.markdown("""
        **How to read this chart:**
        
        - Each point represents a portfolio configuration with its corresponding **expected volatility** (risk) on the x-axis and **expected return** on the y-axis.
        - The **color gradient** indicates the Sharpe Ratio, with darker colors showing portfolios with better risk-adjusted returns.
        - The **red star** represents the portfolio with the highest Sharpe Ratio, which balances return and risk most effectively. Investors might consider this portfolio as an optimal choice.
        """)
    
        # Display Pie chart for the best portfolio allocation
        fig_pie = go.Figure(data=[go.Pie(labels=selected_stocks, values=optimal_weights.round(3)*100)])
        fig_pie.update_layout(title="Best Portfolio Allocation", showlegend=True)
        st.plotly_chart(fig_pie)

        # Add an explanation for the Allocation Pie Chart
        st.markdown("""
        **What this allocation means:**
        
        The pie chart shows the recommended allocation of your investment across the selected stocks to achieve the optimal portfolio. For example:
        
        - If you want to invest **USD 1,000**, multiply the allocation percentage for each stock by USD 1,000. 
          - For instance, if a stock has a 30% allocation, you should invest **USD 300** in that stock.
        - This allocation is based on maximizing the Sharpe Ratio, which provides the best balance between risk and return.
        """)

with tab2:
    st.title("🔮 Forecast Stock Prices Using LSTM")

    # Create a dropdown menu for the user to select the dataset
    selected_ticker = st.selectbox(
        "🪙 Select Stock Dataset for Prediction",
        ["AAPL", "GOOGL", "NKE", "IBM", "JNJ", "KO", "MSFT", "NFLX"]
    )

    # Create a slider for selecting the number of days to predict
    days_to_predict = st.slider(
        "📅 Select Number of Days to Predict",
        min_value=1, 
        max_value=30,   
        value=5,       # Default value 
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
        st.write(data['Adj Close'].tail(days_to_predict))  # Display last n historical prices

 # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot only the predicted prices
    ax.plot(prediction_dates, predicted_prices, linestyle='-', marker='o', color='red', label='Predicted Adj Close')
    
    # Add title and labels
    ax.set_title(f"{selected_ticker} Adjusted Close Price Prediction for Next {days_to_predict} Days")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    
    # Add legend
    ax.legend()
    
    # Set x-axis ticks to exactly match prediction dates
    ax.set_xticks(prediction_dates)
    ax.set_xticklabels(prediction_dates.strftime('%Y-%m-%d'), rotation=45)
    
    # Display the plot in Streamlit
    st.pyplot(fig)   


    # Add an explanation of LSTM and its usage
    st.markdown("""
    ### What is LSTM and How Is It Used Here?
    
    **Long Short-Term Memory (LSTM)** is a type of Recurrent Neural Network (RNN) specifically designed to handle sequential data and long-term dependencies. It is particularly effective in time series forecasting, where the data points are connected over time.
    
    In this app:
    - The **LSTM model** is trained on past stock prices (adjusted close prices) to learn patterns and trends in the data.
    - Using the most recent **60 days of stock prices** as input, the model predicts the next day's price. This process is repeated iteratively to forecast multiple future days.
    
    ### How the Forecast Works:
    1. **Data Preparation**: The historical adjusted close prices are normalized (scaled to a range between 0 and 1) to help the model process the data effectively.
    2. **Sliding Window Input**: The model uses the last 60 days of prices to predict the next day’s price. This sliding window approach ensures the model has context for recent trends.
    3. **Iterative Prediction**: Once the next price is predicted, it is added back to the input data, and the process repeats for the specified number of days.
    
    ### Benefits of LSTM:
    - LSTMs can capture **non-linear relationships** and dependencies in the data, which makes them ideal for forecasting stock prices.
    - The iterative prediction process allows the model to forecast multiple steps into the future.
    
    **Note**: While LSTM models are powerful, stock prices are influenced by numerous external factors (e.g., market conditions, news, and macroeconomic trends) that the model may not account for, so predictions should be used cautiously.
    """)
