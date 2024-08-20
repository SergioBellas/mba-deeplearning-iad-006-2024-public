# Use an official Python image as the base
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

# Expose the port
EXPOSE 8000

# Run the command to start the development server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
#Especificar as Dependências
flask==2.3.2
scikit-learn==1.2.0

#Configurar o Arquivo Principal do Flask (main.py)
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Carregar o modelo de árvore de decisão
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

#Construir a Imagem do Container
docker build -t decision-tree-model .

#Executar o Container
docker run -p 8000:8000 decision-tree-model

#Testar a API
curl -X POST -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}' http://localhost:8000/predict
