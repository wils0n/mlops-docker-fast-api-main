import traceback
import joblib
import logging

# Configura el nivel de registro
logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from fastapi import Request
import numpy as np

model = joblib.load('src/model.joblib')

class_names = np.array(['setosa', 'versicolor', 'virginica'])

app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Iris model API'}

@app.post('/predict')
def predict(data: dict):
    """
    Predicts the class of a given set of features.

    Args:
        data (dict): A dictionary containing the features to predict.
        e.g. {"features": [1, 2, 3, 4]}

    Returns:
        dict: A dictionary containing the predicted class.
    """
    logging.info(f"Datos recibidos: {data}")
    try:
        features = np.array(data['features']).reshape(1, -1)
        logging.info(f"Características reformateadas: {features}")
        
        if model is None:
            logging.error("Modelo no cargado.")
            return {"error": "Modelo no cargado"}, 500
        
        prediction = model.predict(features)
        class_name = class_names[prediction][0]
        return {'predicted_class': class_name}
    except Exception as e:
        logging.error(f"Error en la predicción: {str(e)}")
        return {"error": "Prediction failed"}, 500


@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    logging.error("Unhandled error: %s", traceback.format_exc())
    return PlainTextResponse(str(exc), status_code=500)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logging.error("Validation error: %s", exc.errors())
    return PlainTextResponse(str(exc), status_code=422)

