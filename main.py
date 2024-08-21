from fastapi import FastAPI, File
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pickle
import warnings

import base64
from PIL import Image
import io

warnings.simplefilter(action='ignore', category=DeprecationWarning)

app = FastAPI()

# Definicao dos tipos de dados
class PredictionResponse(BaseModel):
    prediction: float

class ImageRequest(BaseModel):
    image: str

# Carregamento do Modelo de Machine Learning
def load_model():
    global xgb_model_carregado
    with open("xgb_model.pkl", "rb") as f:
        xgb_model_carregado = pickle.load(f)

# Inicializacao da Aplicacao
@app.on_event("startup")
async def startup_event():
    load_model()

# Definicao do endpoint /predict que aceita as requisicoes via POST
# Esse endpoint que ira receber a imagem em base64 e ira converte-la para fazer inferencia
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    # Processamento da Imagem
    img_bytes = base64.b64decode(request.image)
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((8, 8))
    img_array = np.array(img)

    # Converter a imagem pra escala de cinza
    img_array = img_array.reshape(1, -1)

    img_array = img_array.reshape(1, -1)
    
    # Predicao do modelo de Machine Learning
    prediction = xgb_model_carregado.predict(img_array)

    return {"prediction": prediction}
