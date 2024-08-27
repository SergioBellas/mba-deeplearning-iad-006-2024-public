from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
import pickle
from PIL import Image
import io
import base64

app = Flask(giuliana)

# Carregamento do Modelo de Machine Learning
def load_model():
    global xgb_model_carregado
    with open("xgb_model.pkl", "rb") as f:
        xgb_model_carregado = pickle.load(f)

# Carregar o modelo ao iniciar a aplicação
load_model()

# Definição do endpoint /predict que aceita requisições via POST
@app.route('/predict', methods=['POST'])
def predict():
    # Receber a imagem em base64 da requisição
    data = request.get_json()
    img_bytes = base64.b64decode(data['image'])
    
    # Processamento da Imagem
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((8, 8))  # Tamanho da imagem do conjunto de dados digits
    img = img.convert('L')  # Converter para escala de cinza
    img_array = np.array(img).reshape(1, -1)  # Reshape para o formato de entrada do modelo
    
    # Predição do modelo de Machine Learning
    prediction = xgb_model_carregado.predict(img_array)
    
    # Retornar o resultado da predição como JSON
    return jsonify({"prediction": prediction.tolist()})

# Executa a aplicação
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

