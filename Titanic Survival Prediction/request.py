import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Pclass':3, 'Age':22, 'Ftotal':2,'sex':1,'embarked':2})

print(r.json())