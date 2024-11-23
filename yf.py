import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. 주식 데이터 다운로드
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# 2. 주식 데이터 시각화
def visualize_stock_data(data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label=f'{ticker} Close Price', color='blue')
    plt.title(f'{ticker} Closing Price History')
    plt.xlabel('Date')
    plt.ylabel('Price (₩)')
    plt.legend()
    plt.show()

# 3. 주식 데이터 통계 정보 계산 및 출력
def stock_statistics(data, ticker):
    # 통계 계산
    avg_price = data['Close'].mean()
    max_price = data['Close'].max()
    min_price = data['Close'].min()
    max_volume = data['Volume'].max()

    # 첫 번째와 마지막 종가를 .iloc로 가져오기
    first_price = data['Close'].iloc[0]
    last_price = data['Close'].iloc[-1]
    price_change_rate = ((last_price - first_price) / first_price) * 100
    
    # 결과 출력
    print(f"\n--- {ticker} 통계 ---")
    print(f"평균 종가: {avg_price:.2f} ₩")
    print(f"최고 종가: {max_price:.2f} ₩")
    print(f"최저 종가: {min_price:.2f} ₩")
    print(f"최고 거래량: {max_volume}")
    print(f"가격 변동률: {price_change_rate:.2f}%")

# 4. 주식 데이터로 AI 예측
def predict_stock_price(data):
    # 날짜를 숫자로 변환하여 X 데이터로 사용
    data['DateIndex'] = np.arange(len(data))
    X = data['DateIndex'].values.reshape(-1, 1)
    y = data['Close'].values
    
    # 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 선형 회귀 모델 학습
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date Index')
    plt.ylabel('Price (₩)')
    plt.legend()
    plt.show()
    
    # 모델 성능 확인
    print(f"Model R^2 Score: {model.score(X_test, y_test):.2f}")

if __name__ == "__main__":
    # 분석할 주식 종목 리스트
    tickers = ["005930.KS", "000660.KS", "035420.KQ"]  # 삼성전자, SK하이닉스, 네이버
    start_date = "2023-01-01"
    end_date = "2024-11-22"
    
    for ticker in tickers:
        print(f"\n--- {ticker} 데이터 분석 시작 ---")
        try:
            # 데이터 가져오기
            stock_data = get_stock_data(ticker, start_date, end_date)
            
            if stock_data.empty:
                print(f"{ticker}에 대한 데이터를 찾을 수 없습니다.")
            else:
                # 통계 정보 출력
                stock_statistics(stock_data, ticker)
                
                # 시각화
                visualize_stock_data(stock_data, ticker)
                
                # AI 예측
                predict_stock_price(stock_data)
        except Exception as e:
            print(f"{ticker} 분석 중 에러 발생: {e}")
