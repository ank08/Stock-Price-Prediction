import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
# from datetime import date
# # from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
from keras.layers import Dense
from keras import Sequential
import requests


st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Predictor App. Choose your options from below mentioned')
def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()

 

@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df



option = st.sidebar.text_input('Enter a Stock Symbol', value='TSLA')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
        download_data(option, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')




data = download_data(option, start_date, end_date)
scaler = StandardScaler()

def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dataframe
    bb = bb[['Close', 'bb_h', 'bb_l']]
    # MACD
    macd = MACD(data.Close).macd()
    # RSI
    rsi = RSIIndicator(data.Close).rsi()
    # SMA
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()

    if option == 'Close':
        st.write('Close Price')
        st.line_chart(data.Close)
    elif option == 'BB':
        st.write('BollingerBands')
        st.line_chart(bb)
    elif option == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Indicator')
        st.line_chart(rsi)
    elif option == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Expoenetial Moving Average')
        st.line_chart(ema)


def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))



def predict():
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'ANN'])
    tfb=st.radio('Choose an indicator',['Close','Open','High','Low'])
    num = st.number_input('How many days forecast?', value=5)
    num = int(num)
    if st.button('Predict'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num,tfb)
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num,tfb)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num, tfb)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num, tfb)
        elif model == 'ANN':
            ann(num,tfb)

def sentiment():
    sdate = datetime.datetime.strptime(str(start_date), '%Y-%m-%d')
    formatted_sdate = sdate.strftime('%Y%m%dT%H%M')
    endate=datetime.datetime.strptime(str(end_date), '%Y-%m-%d')
    formatted_enddate=endate.strftime('%Y%m%dT%H%M')
    url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers='+ option + '&time_from='+ formatted_sdate +'&time_to='+formatted_enddate+'&apikey=MZ4Q0EB9NUQUEC9Y'
    r = requests.get(url)
    data = r.json()
    ls = []
    print(data)

    for item in data['feed']:
        for ticker_sentiment in item['ticker_sentiment']:
            if ticker_sentiment['ticker'] == 'TSLA':
                ls.append({
                    'time_published': item['time_published'],
                    'ticker_sentiment_score': float(ticker_sentiment['ticker_sentiment_score'])
                })

    # Creating DataFrame
    df = pd.DataFrame(ls)
    avg=df['ticker_sentiment_score'].mean()
    st.write('Sentiment Score:',avg)
    if avg<=-0.35:
        st.write('According to the sentiment the market sof the stock is: <span style="color: green;">Bearish</span>',unsafe_allow_html=True)
    elif avg>-0.35 and avg <= -0.15:
        st.write('According to the sentiment the market of the stock is: <span style="color: green;">Somewhat-Bearish</span>',unsafe_allow_html=True)   

    elif avg > -0.15 and avg < 0.15:
        st.write('According to the sentiment the market of the stock is: <span style="color: green;">Neutral</span>',unsafe_allow_html=True)

    elif 0.15 <= avg < 0.35:
        st.write('According to the sentiment the market of the stock is: Somewhat-Bullish')

    elif avg>=0.35:
        st.write('According to the sentiment the market of the stock is: <span style="color: green;">Bullish</span>',unsafe_allow_html=True)             
def ann(num,indicator):
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = data[[indicator]]
        # shifting the closing price based on number of days forecast
    df['preds'] = data[indicator].shift(-num)
        # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
        # storing the last num_days data
    x_forecast = x[-num:]
        # selecting the required values for training
    x = x[:-num]
        # getting the preds column
    y = df.preds.values
        # selecting the required values for training
    y = y[:-num]
    x = x.reshape(-1, 1)
        #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # Create a Sequential model


    model = Sequential()
    
    # Add the input layer and the first hidden layer
    model.add(Dense(units=128, activation='relu', input_dim=x_train.shape[1] ))
    
    # Add additional hidden layers
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    
    # Add the output layer
    model.add(Dense(units=1)) 
    loss='mae'
    optimizer='adam'
    model.compile(loss=loss,optimizer=optimizer,metrics=['mae'])
    histoy=model.fit(x_train,y_train,validation_data=[x_test,y_test],epochs=50)

    preds = model.predict(x_test)
    sentiment()
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1

def model_engine(model, num,indicator):
    # getting only the closing price
    df = data[[indicator]]
    # shifting the closing price based on number of days forecast
    df['preds'] = data[indicator].shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    #sentiment()
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1


if __name__ == '__main__':
    main()


