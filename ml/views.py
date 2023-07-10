from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from rest_framework.decorators import api_view
from PIL import Image
import os
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from keras.models import load_model
from tensorflow.keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import io
import requests


# @api_view(['POST'])
# def predict(request):
#     # model = load_model(r'F:\full-stack-python\graduation\graduation_project\src\ml\models\ergot.h5')
    
#     model_path = os.path.join(os.getcwd(), 'ml', 'models', 'ergot.h5')
#     model = load_model(model_path)
    
#     image_file = request.FILES['image']
#     image = Image.open(image_file).convert('RGB')
#     image = image.resize((224, 224))
#     image_array = np.array(image) / 255.0
#     image_array = np.expand_dims(image_array, axis=0)

#     prediction = model.predict(image_array)
#     max_prediction = np.argmax(prediction)

#     if max_prediction == 1:
#         result = 'unhealthy'
#     else:
#         result = 'healthy'
#     print(f"<{max_prediction}>")
#     return Response({'result': result})


@api_view(['POST'])
def predict(request):
    model_path = os.path.join(os.getcwd(), 'ml', 'models', 'ergot.h5')
    model = load_model(model_path)
    
    if 'image_url' not in request.data:
        return Response({'error': 'Image URL not provided'}, status=status.HTTP_400_BAD_REQUEST)
    
    image_url = request.data['image_url']
    response = requests.get(image_url)
    
    if response.status_code != 200:
        return Response({'error': 'Failed to fetch image'}, status=status.HTTP_400_BAD_REQUEST)
    
    image = Image.open(io.BytesIO(response.content)).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    max_prediction = np.argmax(prediction)

    if max_prediction == 1:
        result = 'unhealthy'
    else:
        result = 'healthy'
    
    print(f"<{max_prediction}>")
    return Response({'result': result})

# model = load_model(r'F:\full-stack-python\graduation\graduation_project\src\ml\models\dataset.h5')
model_path = os.path.join(os.getcwd(), 'ml', 'models', 'dataset.h5')
model = load_model(model_path)
class FruitClassificationAPI(APIView):
    def post(self, request):
        image_file = request.FILES['image']
        image = Image.open(image_file)
        image = image.resize((224, 224))  # Resize the image to the desired size
        image = image_utils.img_to_array(image)
        image = image.reshape(1, 224, 224, 3)
        image = preprocess_input(image)
        preds = model.predict(image)
        if preds <= 0.5:
            result = "It's Fresh! Eat ahead."
        else:
            result = "It's Rotten. I don't recommend!"
        return Response({"result": result})
