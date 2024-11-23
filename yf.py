import sys
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QComboBox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib

# Matplotlib 백엔드 설정 (Qt5Agg 사용)
matplotlib.use('Qt5Agg')


# 1. 주식 데이터 다운로드
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data


# 2. 주식 데이터 시각화
def visualize_stock_data(data, ticker):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Close'], label=f'{ticker} Close Price', color='blue')
    ax.set_title(f'{ticker} Closing Price History')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (₩)')
    ax.legend()

    return fig


# 3. 주식 데이터 통계 정보 계산 및 출력
def stock_statistics(data, ticker):
    avg_price = data['Close'].mean()
    max_price = data['Close'].max()
    min_price = data['Close'].min()
    max_volume = data['Volume'].max()
    first_price = data['Close'].iloc[0]
    last_price = data['Close'].iloc[-1]
    price_change_rate = ((last_price - first_price) / first_price) * 100

    stats = (
        f"--- {ticker} 통계 ---\n"
        f"평균 종가: {avg_price:.2f} ₩\n"
        f"최고 종가: {max_price:.2f} ₩\n"
        f"최저 종가: {min_price:.2f} ₩\n"
        f"최고 거래량: {max_volume}\n"
        f"가격 변동률: {price_change_rate:.2f}%"
    )
    return stats


# 4. 주식 데이터로 AI 예측
def predict_stock_price(data, ticker):
    data['DateIndex'] = np.arange(len(data))
    X = data['DateIndex'].values.reshape(-1, 1)
    y = data['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(X_test, y_test, color='blue', label='Actual Prices')
    ax.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices')
    ax.set_title(f'{ticker} Stock Price Prediction')
    ax.set_xlabel('Date Index')
    ax.set_ylabel('Price (₩)')
    ax.legend()

    return fig, f"Model R^2 Score: {model.score(X_test, y_test):.2f}"


# PyQt5 GUI
class StockAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Analyzer")
        self.setGeometry(100, 100, 500, 400)

        # Main layout
        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)

        # Label
        self.label = QLabel("분석할 주식을 선택하세요:")
        self.layout.addWidget(self.label)

        # Dropdown
        self.ticker_dropdown = QComboBox()
        self.tickers = ["005930.KS", "000660.KS", "035420.KQ"]  # 삼성전자, SK하이닉스, 네이버
        self.ticker_dropdown.addItems(self.tickers)
        self.layout.addWidget(self.ticker_dropdown)

        # Buttons
        self.analyze_button = QPushButton("분석 및 시각화")
        self.analyze_button.clicked.connect(self.analyze_stock)
        self.layout.addWidget(self.analyze_button)

        # Label for results
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.layout.addWidget(self.result_label)

        # 저장된 그래프를 표시할 레이아웃
        self.graph_layout = QVBoxLayout()
        self.layout.addLayout(self.graph_layout)

        self.setCentralWidget(self.central_widget)

    def analyze_stock(self):
        # 이전 그래프 제거
        for i in reversed(range(self.graph_layout.count())):
            widget = self.graph_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        ticker = self.ticker_dropdown.currentText()
        start_date = "2023-01-01"
        end_date = "2024-11-22"

        try:
            stock_data = get_stock_data(ticker, start_date, end_date)

            if stock_data.empty:
                self.result_label.setText(f"{ticker}에 대한 데이터를 찾을 수 없습니다.")
            else:
                # 통계 정보
                stats = stock_statistics(stock_data, ticker)
                self.result_label.setText(stats)

                # 시각화
                fig = visualize_stock_data(stock_data, ticker)
                canvas = FigureCanvas(fig)  # PyQt5와 연결
                self.graph_layout.addWidget(canvas)

                # AI 예측
                fig_pred, prediction_score = predict_stock_price(stock_data, ticker)
                canvas_pred = FigureCanvas(fig_pred)
                self.graph_layout.addWidget(canvas_pred)

                self.result_label.setText(f"{stats}\n\n{prediction_score}")
        except Exception as e:
            self.result_label.setText(f"{ticker} 분석 중 에러 발생: {e}")


# PyQt 앱 실행
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = StockAnalyzerApp()
    main_window.show()
    sys.exit(app.exec_())
