<h1>Stock Prediction and Portfolio Optimization App</h1>

<h2>Overview</h2>
<p>This Streamlit app integrates advanced machine learning models and financial techniques to provide investors and analysts with insights into stock prices and portfolio management. The app enables users to explore stock data, predict future stock prices, and optimize portfolio allocations using Monte Carlo simulations. It’s a comprehensive tool aimed at empowering users to make smarter, data-driven investment decisions.</p>

<h2>Features</h2>
<ul>
  <li><strong>Stock Exploration:</strong> Analyze historical stock performance, visualize trends, and view company-specific metrics like market cap, average volume, and 52-week highs/lows.</li>
  <li><strong>Price Prediction:</strong> Predict up to 30 days of stock prices using pretrained LSTM models for popular stocks like Apple, Google, and Microsoft.</li>
  <li><strong>Portfolio Optimization:</strong> Optimize portfolio allocations using Monte Carlo simulations to maximize the Sharpe Ratio and balance risk vs. return.</li>
</ul>

<h2>Project Structure</h2>
<pre>
stock_prediction_portfolio_optimization/
│
├── app.py                              # Main Streamlit app script
├── requirements.txt                    # Dependencies to run the app
├── README.md                           # Documentation of the project
├── Data_Collection_&_Feature_Engineering_(msft).ipynb  # Data collection and feature engineering notebook
├── ML_Models.ipynb                     # Machine learning model training and testing
├── Portfolio Optimization Using Monte Carlo Simulations (1).ipynb  # Portfolio optimization notebook
├── models/
│   ├── lstm_Apple.keras                # Pretrained LSTM model for Apple
│   ├── lstm_google.keras               # Pretrained LSTM model for Google
│   ├── lstm_ibm.keras                  # Pretrained LSTM model for IBM
│   ├── xgboost_model_google.pkl        # Pretrained XGBoost model for Google
│   └── ...                             # Other pretrained models
└── data/
    └── historical_stock_data.csv       # Processed historical stock data
</pre>

<h2>Setup & Installation</h2>
<ol>
  <li><strong>Clone the Repository:</strong>
    <pre>git clone https://github.com/your-username/stock_prediction_portfolio_optimization.git
cd stock_prediction_portfolio_optimization</pre>
  </li>
  <li><strong>Create a Virtual Environment (Optional but recommended):</strong>
    <pre>python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`</pre>
  </li>
  <li><strong>Install Dependencies:</strong>
    <pre>pip install -r requirements.txt</pre>
  </li>
  <li><strong>Run the Application:</strong>
    <pre>streamlit run app.py</pre>
  </li>
</ol>

<h2>Usage</h2>
<p>After running the app, visit <a href="http://localhost:8501">http://localhost:8501</a> in your web browser to access the app. Follow the on-screen instructions to explore stock data, predict prices, and optimize your portfolio.</p>

<h2>Data Sources</h2>
<ul>
  <li>Historical stock price data provided by <a href="https://finance.yahoo.com/">Yahoo Finance API</a></li>
</ul>

<h2>Author</h2>
<p>This project was created by <strong>David Heller</strong>, a data scientist with expertise in finance and machine learning. Connect with me:</p>
<ul>
  <li><a href="https://www.linkedin.com/in/david-heller-w/">LinkedIn</a></li>
  <li><a href="https://github.com/davidhellerw">GitHub</a></li>
  <li><a href="https://davidhellerw.com/">Personal Website</a></li>
</ul>

<h2>License</h2>
<p>This project is licensed under the <a href="LICENSE">MIT License</a>.</p>

<p>Happy investing and learning! 🎉</p>
