# S&P 500 and AAPL Stock Price Prediction Model 
### Created by Sean McDermott, Haran Eiger, and Jon Cook.

# Required data sets:
> https://pypi.org/project/yfinance/

> https://www.kaggle.com/datasets/lorilaz/apple-news-headline-sentiment-and-stock-info 

# Introduction
The stock market is defined by its non-linear data patterns and its unpredictability. Despite its complexities and the widely accepted Efficient Market Hypothesis, which claims that it is almost impossible to consistently outperform the market without insider information, there is substantial value in developing models that can predict market movements. This project focuses on using various financial indicators to predict the S&P 500 change in magnitude and direction.<br><br>

Our project aimed to utilize deep learning to forecast stock movements. We chose to focus on a historical dataset of the S&P 500, spanning from 2010 to 2022, which we accessed using the yfinance library. This dataset allowed us to dive deep into over a decade of stock behavior during various economic conditions. The core of our approach involved utilizing the indicators calculated through the data set. We calculated indicators such as the Relative Strength Index (RSI) Exponential Moving Averages (EMA) and Bollinger Bands (BB) over different time frames. These indicators help identify momentum and trends in the stock market, providing signals that can indicate potential future movements. To make our data suitable for a deep learning model, we engineered additional features that represent daily price changes and set up binary flags indicating whether the stock price was likely to rise or fall the following day. This transformation was crucial for preparing our dataset for the next stage of our process.<br><br>

Using the MinMaxScaler, we normalized our features to ensure they were well-suited for modeling, which eliminated bias that could arise from the different scales of raw financial data. We then organized this data into 30-day sequences, each serving as an individual sample for our model. This structuring is key for Long Short-Term Memory (LSTM), which thrives on finding patterns in time-series data.
Our model's architecture features an LSTM layer with 150 units, designed to decode the temporal dependencies within stock prices. This is followed by a dense output layer that employs a linear activation function to accurately map the LSTM outputs to our prediction targets. We compiled the model using the Adam optimizer and mean squared error loss function to optimize performance. The training was conducted over 30 epochs, with a split of the data—80% for training and 20% for validation. This ensures that the model avoids overfitting, enabling it to generalize unseen data.<br>

# Initial Approach: Binary Classification Using Machine Learning
After analyzing our indicators and preparing our data for analysis we examined which neural networks would perform best at classifying whether the market would be up or down on the following day. Early on in our research, we wanted to see how simple models would perform and then move on to more complex models to see if they could fit the entanglements of the stock market well.<br><br>

The first model we looked into was a feed-forward neural network with three dense layers. The first two layers had 64 nodes each and reLU as the activation function. The final layer had only two nodes for the binary classification and an activation of softmax. Then, we decided to perform dropout(0.5) after each of the dense layers with 64 nodes to intentionally “damage” the network so that it does not overfit the data. This simple model led to an accuracy of **0.5529**.<br><br>

The second model we decided to use for stock price prediction was an RNN or recurrent neural network. A simple RNN was expected to perform better than a feed-forward neural network because the stock data is sequential. RNNs apply weights recursively to the input to obtain a prediction for sequential data. We employed the same basic structure for the RNN as we did for the feed-forward: two model layers, between dropout(0.5) layers, and a dense layer with two nodes for classification (up or down). The results of this prediction were outstanding high, having a test accuracy of **0.5952**.<br><br>

The third model we trained on our S&P 500 stock price data was an LSTM. Long Short term memory (LSTM) models have been considered the most useful in predicting the market because of their ability to work with time series data. LSTMs are said to be able to pick up trends and identify patterns that RNNs cannot because do not handle long-term dependencies well. With data that extends over a decade, an LSTM model seemed like the clear choice. However, with the same basic structure as the previous two models, the LSTM only had an accuracy of **0.5786**.<br><br>

**The RNN outperformed the advanced LSTM model due to the structure of the data**. For each prediction, we only looked at a certain number of days or “back candles” in the past. We found that the optimal number of days to look back was 15 or three weeks. The number of days is considered short-term, thus the RNN predicted the price better because it handles short-term dependencies better.<br> 

# Second Approach: Predicting the Magnitude of Change for a Single Stock Using Sentiment Analysis
### Introduction
Utilizing a combination of sentiment analysis and traditional financial indicators, this approach aims to predict the magnitude of price changes in Apple's stock. Sentiment analysis, which evaluates the tone and context of text data—specifically, news headlines in this case—provides insights into the general market sentiment at any given time. By integrating these sentiment scores with quantitative financial data such as the Relative Strength Index (RSI) and Exponential Moving Averages (EMA), we seek to construct a nuanced model that can better anticipate stock price fluctuations.<br><br>

The rationale behind combining these diverse datasets stems from the hypothesis that market sentiment can often precede or amplify market movements, particularly in a company as prominent as Apple. News and public perception can significantly influence investor behavior, thereby affecting stock prices. Our approach leverages historical data from Yahoo Finance and sentiment data derived from news headlines, aligning them by date to create a robust dataset for our analysis.<br>
### Apple Sentiment Analysis Data


