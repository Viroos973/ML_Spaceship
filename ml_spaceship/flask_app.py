from flask import Flask, request, jsonify
from model import My_Classifier_Model

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    dataset_path = request.args.get('dataset')
    model_instance = My_Classifier_Model(dataset=dataset_path)
    
    model_instance.train()
    return jsonify({'message': 'Модель успешно обучена!'})

@app.route('/predict', methods=['POST'])
def predict_results():
    dataset_path = request.args.get('dataset')
    model_instance = My_Classifier_Model(dataset=dataset_path)
    
    model_instance.predict()
    return jsonify({'message': 'Результаты успешно предсказаны и сохранены в файл results.csv'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)