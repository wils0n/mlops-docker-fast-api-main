import json
import requests

data = [[4.3, 3. , 1.1, 0.1],
       [5.8, 4. , 1.2, 0.2],
       [5.7, 4.4, 1.5, 0.4],
       [5.4, 3.9, 1.3, 0.4],
       [5.1, 3.5, 1.4, 0.3],
       [5.7, 3.8, 1.7, 0.3],
       [5.1, 3.8, 1.5, 0.3],
       [5.4, 3.4, 1.7, 0.2],
       [5.1, 3.7, 1.5, 0.4],
       [4.6, 3.6, 1. , 0.2],
       [5.1, 3.3, 1.7, 0.5],
       [4.8, 3.4, 1.9, 0.2]]

url = 'http://0.0.0.0:8000/predict/'
#url = 'https://servicio-api-mlops9-2-kp4nj3muxq-ue.a.run.app/predict/'


predictions = []
for record in data:
    payload = {'features': record}
    payload = json.dumps(payload)
    
    # Hacer la solicitud POST
    response = requests.post(url, data=payload, headers={'Content-Type': 'application/json'})
    
    # Imprimir la respuesta para ver el código de estado y el cuerpo
    print(f"Response Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print(f"Response JSON: {response.json()}")
        predictions.append(response.json()['predicted_class'])
    else:
        print(f"Error Response: {response.text}")
    
print("Predictions:", predictions)
