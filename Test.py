import sys
import numpy as np
import pandas as pd
sys.path.append("./utils/")
import Main as mm
import Post_analysis as pa

# 데이터 로드
data = pd.read_csv("./data/movie_data.csv")

# 모델 인스턴스 생성
main = mm.learning()

# 데이터 전처리
main.preprocess(data, input_col='document',version="bert")

# bert 모델 로드
main.select_model(selected_model="Bert_classification")

# 모델 학습
main.fit(test_ratio=0.1, batch_size=512, EPOCHS=1)

# 모델 예측
main.predict(" 그 모델은 별로다")

main.predict(["그 모델은 별로다"], ["어구어구 어구구"])

# 모델 사후 분석
# 라임 분석
main.lime(" 그 모델은 별로다")
 
# binary classification 분석
import Post_analysis as pa
pred = main.predict(data["document"])
pa.report(pred=pred,labels= data["label"])

# 모델 저장
main.save("./save_model")

# 모델 로드
main.load("./save_model")
