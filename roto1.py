import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# CSV 파일에서 데이터 불러오기
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

# LSTM 모델 구축
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(6))  # 로또 번호 6개를 출력

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# 모델 평가
test_loss = model.evaluate(X_test, y_test)
print(f"LSTM 모델의 테스트 손실: {test_loss}")

# 확률이 높은 번호를 예측하는 함수
def generate_lotto_numbers(model, X_train, scaler, n_top=30):
    predictions = model.predict(X_train)
    
    # 예측된 값들을 1~45 사이로 변환
    predicted_numbers = scaler.inverse_transform(predictions).flatten()

    # 예측된 값들 중에서 1~45 범위로 변환한 후 중복 제거
    predicted_numbers = np.round(predicted_numbers).astype(int)
    predicted_numbers = np.clip(predicted_numbers, 1, 45)

    # 각 번호의 출현 빈도 계산 (중복된 값을 방지하기 위한 처리 추가)
    number_counts = pd.Series(predicted_numbers).value_counts().sort_values(ascending=False)

    # 상위 n_top개의 번호를 선택
    top_numbers = number_counts.index[:n_top].tolist()

    return top_numbers

# 상위 30개의 번호 예측
top_numbers = generate_lotto_numbers(model, X_train, scaler, n_top=30)

# 1순위 ~ 5순위로 나누어 번호 추천
def recommend_lotto_numbers(top_numbers):
    recommended_sets = []
    
    # 각 순위별로 6개씩 나눠서 추천
    for i in range(5):  # 1순위부터 5순위까지
        recommended_set = sorted(top_numbers[i*6:(i+1)*6])
        
        # 만약 중복된 숫자나 부족한 경우 1~45 사이에서 랜덤으로 추가
        while len(recommended_set) < 6:
            random_num = np.random.randint(1, 46)
            if random_num not in recommended_set:
                recommended_set.append(random_num)
                
        recommended_sets.append(sorted(recommended_set))
    
    return recommended_sets

# 1순위 ~ 5순위 추천 번호 출력
recommended_sets = recommend_lotto_numbers(top_numbers)
for i, numbers in enumerate(recommended_sets):
    print(f"{i+1}순위 추천 번호: {numbers}")


