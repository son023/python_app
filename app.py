from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)

# Tải mô hình
mon1_model = tf.keras.models.load_model('mon1_rnn.keras')
mon2_model = tf.keras.models.load_model('mon2_rnn.keras')
def check(x):
    if x>=9.0: return 'A+'
    elif x>=8.5: return 'A'
    elif x>=8: return 'B+'
    elif x>=7: return 'B'
    elif x>=6.5: return 'C+'
    elif x>=6: return 'C'
    elif x>=5: return 'D+'
    elif x>=4: return 'D'
    else: return 'F'
def tb1(a,b,c,d,x):
    if x==1:
        return (a*10+b*10+c*20+d*60)/100
    else: return (a*10+b*20+c*20+d*50)/100

@app.route('/api/predict/rnn', methods=['POST'])
def predict_rnn():
    try:
        data = request.get_json()
        input_data = data['input']
        input_data = [float(i) for i in input_data]
        a = input_data
        if len(input_data) != 3:
            return jsonify({'error': 'Dữ liệu đầu vào phải có 3 điểm'}), 400
        input_data = np.array(input_data).reshape(1, 1, 3)
        prediction = mon1_model.predict(input_data)
        ck = round(prediction.tolist()[0][0], 1)
        print(ck)
        tb = tb1(a[0],a[1],a[2],ck,1)
       
        return jsonify({'prediction': prediction.tolist()[0][0],'diem':tb, 'diemChu':check(tb)}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# API dự đoán với mô hình LSTM
@app.route('/api/predict/lstm', methods=['POST'])
def predict_lstm():
    try:
        data = request.get_json()
        input_data = data['input']  
        a=input_data
        if len(input_data) != 3:
            return jsonify({'error': 'Dữ liệu đầu vào phải có 3 điểm'}), 400
        input_data = np.array(input_data).reshape(1, 1, 3)
        prediction = mon2_model.predict(input_data)
        ck = round(prediction.tolist()[0][0], 1)
        tb = tb1(a[0],a[1],a[2],ck,2)
        return jsonify({'prediction': prediction.tolist()[0][0],'diem':tb, 'diemChu':check(tb)}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/', methods=['GET'])
def ok():
    return '<h1>Xin chao, Son day</h1>'
# Chạy server
if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=8080)
