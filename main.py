from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pickle
from PIL import Image
import io
import base64

app = FastAPI()

# Carregamento do Modelo de Machine Learning
def load_model():
    global xgb_model_carregado
    with open("xgb_model.pkl", "rb") as f:
        xgb_model_carregado = pickle.load(f)

# Carregar o modelo ao iniciar a aplicação
load_model()

# Definição da classe para receber a imagem em base64
class ImageData(BaseModel):
    image: str

# Definição do endpoint /predict que aceita requisições via POST
@app.post('/predict')
async def predict(data: ImageData):
    # Receber a imagem em base64 da requisição
    img_bytes = base64.b64decode(data.image)
    
    # Processamento da Imagem
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((8, 8))  # Tamanho da imagem do conjunto de dados digits
    img = img.convert('L')  # Converter para escala de cinza
    img_array = np.array(img).reshape(1, -1)  # Reshape para o formato de entrada do modelo
    
    # Predição do modelo de Machine Learning
    prediction = xgb_model_carregado.predict(img_array)
    
    # Retornar o resultado da predição como JSON
    return {"prediction": prediction.tolist()}

# Executa a aplicação
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
