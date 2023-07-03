from django.urls import path
from .views import predict,FruitClassificationAPI

urlpatterns = [
    path('classify-wheat/', predict, name='predict_image'),
    path('classify-fruit/', FruitClassificationAPI.as_view(), name='classify'),
    # path('predict2/', predict2, name='predict2_image'),
    
    
]