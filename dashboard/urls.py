from django.urls import path

from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.index, name='index'),
    path('state/<str:sName>', views.state, name='state'),
    path('report/', views.report, name='report'),
    path('forecast/', views.forecast, name='forecast'),
]