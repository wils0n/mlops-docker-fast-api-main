import traceback
import joblib
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from fastapi.exceptions import RequestValidationError

from pydantic import BaseModel
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn.utils.rnn import pad_sequence

# Configura el nivel de registro
logging.basicConfig(level=logging.INFO)

# Inicializar el tokenizador RoBERTa
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Cargar el modelo RoBERTa desde el archivo joblib
model = joblib.load('src/model.joblib')

# Definir el diccionario id_to_etiqueta
id_to_etiqueta = {
    0: 'medium',
    1: 'easy',
    2: 'hard',
    3: 'challenge',
    4: 'beginner'
}

app = FastAPI()

class TextData(BaseModel):
    title: str
    statement: str

@app.get('/')
def read_root():
    return {'message': 'RoBERTa model API'}

@app.post('/predict')
def predict(data: TextData):
    """
    Predicts the class of a given set of text data.

    Args:
        data (TextData): A dictionary containing the title and statement to predict.
        e.g. {
        "title": "backup functions",
        "statement": "all submissions for this problem are available one unavoidable problem with running a restaurant is that occasionally a menu item cannot be prepared this can be caused by a variety of reasons such as missing ingredients or malfunctioning equipment there is an additional problem with such situations a customer who has spent many minutes deciding between menu items can get quite annoyed when they place the order and the server tells them the item is not available to mitigate this effect the chef declares that all servers must have what he calls a backup function this is a function f from menu items to menu items for each menu item x fx is an alternative menu item that the server may suggest to the customer in the event the customer requests menu item x when x is not available of course if a given item is not available then some other items make better suggestions than others so for some pairs of items x and y the chef has determined a numeric value describing the effectiveness of the assignment fx  y higher values indicate the proposed substitute is similar to the original and lower values indicate the proposed substitute is not very similar to the original such effectiveness values are symmetric meaning that if the effectiveness of assignment fx  y is v then the effectiveness of the assignment fy  x is also v you will be given a list of pairs of menu items each such pair will come with an associated effectiveness value you are to compute a backup function f from these pairs however there is one additional constraint for personal reasons the chef is opposed to using two items as backups for each other thus for any two menu items x and y it cannot be that fx  y and fy  x your goal is to compute a backup function of maximum possible quality the quality of the backup function is simply defined as the sum of the effectiveness values of the assignments fa for each item a input the first line contains a single integer t   indicating the number of test cases each test case begins with two integers n and m where   n   and   m   where n is the number of items in the menu the menu items will be numbered  through n m lines follow each containing three integers ab and v here   a  b  n and v   such a triple indicates that fa  b or fb  a but not both may be used in a backup function setting either fa  b or fb  a has effectiveness v each pair   a  b  n will occur at most once in the input test cases will be separated by a single blank line including a blank line preceding the first test case output the output for each test case consists of a single line containing the maximum possible quality of a backup function f that does not assign any pair of items to each other to be explicit the assignment fa  b has effectiveness v where v is the value associated with the ab pair in the input then the quality of a backup function is just the sum of the effectiveness values of the assignment for each item if no backup function can be constructed from the given input pairs then simply output impossible note the lowercase i we may only assign fa  b or fb  a if ab is an input pair furthermore a backup function defines a single item fa for every item a between  and n note that fa  a is not valid since fa is supposed to be an alternative item if menu item a is not available and all input pairs ab have a  b anyway finally based on the chefs additional constraint we cannot have a backup function with both fa  b and fb  a for any input pair ab example"
        }

    Returns:
        dict: A dictionary containing the predicted class.
    """
    logging.info(f"Datos recibidos: {data}")
    try:
        # Tokenizar el texto del título y la declaración
        title_tokens = tokenizer.encode(data.title, add_special_tokens=True)
        statement_tokens = tokenizer.encode(data.statement, add_special_tokens=True)

        # Convertir las listas de tokens a tensores de PyTorch con padding
        title_tensor = pad_sequence([torch.tensor(title_tokens)], batch_first=True, padding_value=0)
        statement_tensor = pad_sequence([torch.tensor(statement_tokens)], batch_first=True, padding_value=0)

        # Evaluar el modelo y obtener predicciones
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=title_tensor, attention_mask=(title_tensor > 0))
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predicted_class = preds.item()
            predicted_level = id_to_etiqueta[predicted_class]

        return {'predicted_class': predicted_class, 'predicted_level': predicted_level}
    except Exception as e:
        logging.error(f"Error en la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
