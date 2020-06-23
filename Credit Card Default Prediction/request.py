import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'balance':255, 'income':529, 'student_Yes':1})

print(r.json())