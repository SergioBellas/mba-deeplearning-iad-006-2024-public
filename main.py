from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pickle
from PIL import Image
import io
import base64

# Criação da instância FastAPI
app = FastAPI()

# Carregamento do Modelo de Machine Learning
def load_model():
    global xgb_model_carregado
    try:
        with open("xgb_model.pkl", "rb") as f:
            xgb_model_carregado = pickle.load(f)
        print("Modelo carregado com sucesso")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        raise e

# Carregar o modelo ao iniciar a aplicação
load_model()

# Definição da classe para receber a imagem em base64
class ImageData(BaseModel):
    image: str

# Definição do endpoint /predict que aceita requisições via POST
@app.post('/predict')
async def predict(data: ImageData):
    try:
        # Receber a imagem em base64 da requisição
        print("Recebendo imagem...")
        img_bytes = base64.b64decode(data.image)
        
        # Processamento da Imagem
        print("Processando imagem...")
        img = Image.open(io.BytesIO(img_bytes))
        img = img.resize((8, 8))  # Tamanho da imagem do conjunto de dados digits
        img = img.convert('L')  # Converter para escala de cinza
        img_array = np.array(img).reshape(1, -1)  # Reshape para o formato de entrada do modelo
        print("Imagem processada com sucesso")
        
        # Predição do modelo de Machine Learning
        print("Realizando predição...")
        prediction = xgb_model_carregado.predict(img_array)
        print("Predição realizada com sucesso")
        
        # Retornar o resultado da predição como JSON
        return {"prediction": prediction.tolist()}
    except Exception as e:
        print(f"Erro ocorrido durante a predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Executa a aplicação
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
