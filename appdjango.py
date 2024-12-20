# settings.py (à ajouter dans votre fichier settings.py existant)
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# urls.py (projet principal)
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('classifier.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# classifier/urls.py
from django.urls import path
from . import views

app_name = 'classifier'

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
]

# classifier/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import time

class FruitClassifier:
    def __init__(self):
        self.model = load_model('model.h5')
        self.class_names = ["apple", "banana", "orange"]

    def predict(self, img_path):
        img = image.load_img(img_path, target_size=(32, 32))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        prediction = self.model.predict(x)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        return self.class_names[predicted_class], confidence

# Initialiser le classificateur
classifier = FruitClassifier()

def index(request):
    return render(request, 'classifier/indexdjango.html')

def predict(request):
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            upload = request.FILES['file']
            
            # Créer un nom de fichier unique
            filename = f"upload_{time.time()}{os.path.splitext(upload.name)[1]}"
            filepath = os.path.join(settings.MEDIA_ROOT, filename)
            
            # Sauvegarder le fichier
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            with open(filepath, 'wb+') as destination:
                for chunk in upload.chunks():
                    destination.write(chunk)
            
            # Faire la prédiction
            predicted_class, confidence = classifier.predict(filepath)
            
            # Nettoyer
            os.remove(filepath)
            
            return JsonResponse({
                'prediction': predicted_class,
                'confidence': float(confidence)
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Invalid request'}, status=400)