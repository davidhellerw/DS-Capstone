<h1>Stock Prediction and Portfolio Optimization App</h1>

<h2>Overview</h2>
<p>The <strong>Stock Prediction and Portfolio Optimization App</strong> is a powerful, interactive Streamlit-based application designed to empower investors and analysts. By combining advanced machine learning models and sophisticated financial techniques, this app helps users analyze stock data, forecast future stock prices, and optimize portfolio allocations. Whether you're a seasoned investor or a beginner, this app provides the insights you need to make smarter, data-driven investment decisions.</p>

<p>Access the app live at: <a href="https://stock-forecast-and-allocate.streamlit.app/">Stock Forecast and Allocate</a></p>

<h2>Features</h2>
<ul>
  <li><strong>Stock Exploration:</strong>
    <ul>
      <li>Analyze historical stock performance with intuitive visualizations.</li>
      <li>Track key metrics such as 52-week high/low, market capitalization, average trading volume, and moving averages (20-day and 50-day).</li>
      <li>Fetch detailed company information, including short ratio, previous close price, and website links.</li>
      <li>Summarize company metrics in a tabular format for easy comparison.</li>
      <li>Customize analysis by selecting specific date ranges.</li>
    </ul>
  </li>
  <li><strong>Price Prediction Using LSTM:</strong>
    <ul>
      <li>Predict up to 30 days of stock prices using pretrained Long Short-Term Memory (LSTM) models.</li>
      <li>Available for popular tickers such as Apple, Google, Microsoft, and more.</li>
      <li>Interactive features include a slider to set prediction days and comparisons of predicted vs. historical prices.</li>
      <li>Visualize predictions with dynamic charts.</li>
    </ul>
  </li>
  <li><strong>Portfolio Optimization:</strong>
    <ul>
      <li>Optimize portfolio allocations using Monte Carlo simulations with over 10,000 iterations.</li>
      <li>Maximize the Sharpe Ratio to balance risk and return effectively.</li>
      <li>Visualize portfolio performance using scatter plots of risk vs. return, color-coded by Sharpe Ratio.</li>
      <li>Generate pie charts to display the best allocation strategy.</li>
      <li>Obtain detailed metrics such as expected return, volatility, and Sharpe Ratio for the optimal portfolio.</li>
    </ul>
  </li>
</ul>

<h2>Tools and Technologies</h2>
<ul>
  <li><strong>Programming Languages:</strong> Python</li>
  <li><strong>Machine Learning Frameworks:</strong> TensorFlow, Keras, Scikit-learn</li>
  <li><strong>Data Handling:</strong> Pandas, NumPy, Yahoo Finance API (yfinance)</li>
  <li><strong>Visualization Tools:</strong> Matplotlib, Plotly, MplFinance</li>
  <li><strong>Portfolio Optimization:</strong> Monte Carlo simulations</li>
  <li><strong>Deployment:</strong> Streamlit Cloud</li>
</ul>

<h2>Project Structure</h2>
<pre>
stock_prediction_portfolio_optimization/
│
├── .devcontainer/                       # Dev container configuration
│   └── devcontainer.json
├── app.py                               # Main Streamlit app script
├── appi.py                              # Alternate app script (for testing and development)
├── requirements.txt                     # Dependencies to run the app
├── README.md                            # Documentation of the project
├── Data_Collection_&_Feature_Engineering_(msft).ipynb  # Data preprocessing notebook
├── ML_Models.ipynb                      # Notebook for machine learning models
├── Portfolio Optimization Using Monte Carlo Simulations (1).ipynb  # Portfolio optimization notebook
├── models/                              # Directory for pretrained models
│   ├── lstm_Apple.keras                 # Pretrained LSTM model for Apple
│   ├── lstm_google.keras                # Pretrained LSTM model for Google
│   ├── lstm_ibm.keras                   # Pretrained LSTM model for IBM
│   ├── lstm_jnj.keras                   # Pretrained LSTM model for Johnson & Johnson
│   ├── lstm_ko.keras                    # Pretrained LSTM model for Coca-Cola
│   ├── lstm_msft.keras                  # Pretrained LSTM model for Microsoft
│   ├── lstm_netflix.keras               # Pretrained LSTM model for Netflix
│   ├── lstm_nike.keras                  # Pretrained LSTM model for Nike
│   ├── xgboost_model_google.pkl         # Pretrained XGBoost model for Google
│   ├── xgboost_model_ibm.pkl            # Pretrained XGBoost model for IBM
│   ├── xgboost_model_jnj.pkl            # Pretrained XGBoost model for Johnson & Johnson
│   ├── xgboost_model_ko.pkl             # Pretrained XGBoost model for Coca-Cola
│   ├── xgboost_model_msft.pkl           # Pretrained XGBoost model for Microsoft
│   └── xgboost_model_netflix.pkl        # Pretrained XGBoost model for Netflix
├── data/                                # Directory for processed data and CSV files
│   ├── APPLE.csv                        # Historical stock data for Apple
│   ├── GOOGLE.csv                       # Historical stock data for Google
│   ├── IBM.csv                          # Historical stock data for IBM
│   ├── JNJ.csv                          # Historical stock data for Johnson & Johnson
│   ├── KO.csv                           # Historical stock data for Coca-Cola
│   ├── MSFT.csv                         # Historical stock data for Microsoft
│   ├── NETFLIX.csv                      # Historical stock data for Netflix
│   └── NIKE.csv                         # Historical stock data for Nike
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
