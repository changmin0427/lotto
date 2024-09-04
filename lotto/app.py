from flask import Flask, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

app = Flask(__name__)

# CSV 파일에서 데이터 불러오기 (백엔드 서버에서 동작)
data = pd.read_csv('lotto_data.csv', header=None)

# 데이터 배열로 변환
X = data.values

# 데이터 전처리: MinMaxScaler를 사용해 데이터를 0과 1 사이로 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# 시퀀스 데이터 생성 함수
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

# 시퀀스 길이 설정
sequence_length = 5

# 시퀀스 데이터 생성
X_seq = create_sequences(X_scaled, sequence_length)

# 타겟 값은 각 회차의 다음 회차에 해당하는 번호로 설정
y = X[sequence_length:]

# 데이터 분할 (train/test split)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)

# LSTM 모델 생성
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(6))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# 확률이 높은 번호를 예측하는 함수
def generate_lotto_numbers():
    predictions = model.predict(X_train)
    predicted_numbers = scaler.inverse_transform(predictions).flatten()
    predicted_numbers = np.round(predicted_numbers).astype(int)
    predicted_numbers = np.clip(predicted_numbers, 1, 45)
    number_counts = pd.Series(predicted_numbers).value_counts().sort_values(ascending=False)
    top_numbers = number_counts.index[:30].tolist()

    recommended_sets = []
    for i in range(5):
        recommended_set = sorted(top_numbers[i*6:(i+1)*6])
        while len(recommended_set) < 6:
            random_num = np.random.randint(1, 46)
            if random_num not in recommended_set:
                recommended_set.append(random_num)
        recommended_sets.append(sorted(recommended_set))

    return recommended_sets

# API 엔드포인트 생성
@app.route('/predict', methods=['GET'])
def predict():
    recommended_sets = generate_lotto_numbers()
    return jsonify(recommended_sets)

# 웹페이지 렌더링
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)