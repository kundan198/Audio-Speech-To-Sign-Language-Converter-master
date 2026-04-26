"""A2SL URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('about/',views.about_view,name='about'),
    path('contact/',views.contact_view,name='contact'),
    path('login/',views.login_view,name='login'),
    path('logout/',views.logout_view,name='logout'),
    path('signup/',views.signup_view,name='signup'),
    path('animation/',views.animation_view,name='animation'),
    path('sign-to-text/', views.sign_to_text_view, name='sign_to_text'),
    path('conversation/', views.conversation_view, name='conversation'),
    path('api/recognize-sign/', views.recognize_sign_view, name='recognize_sign'),
    path('api/simplify-text/', views.simplify_text_view, name='simplify_text'),
    path('api/elevenlabs/voices/', views.elevenlabs_voices_view, name='elevenlabs_voices'),
    path('api/elevenlabs/tts/', views.elevenlabs_tts_view, name='elevenlabs_tts'),
    path('api/formulate-sentence/', views.formulate_sentence_view, name='formulate_sentence'),
    path('api/classify-hand/', views.classify_hand_view, name='classify_hand'),
    path('api/translate-signs/', views.translate_signs_view, name='translate_signs'),
    path('',views.home_view,name='home'),
]
