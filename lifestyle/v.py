from django.shortcuts import render
from django.http import HttpResponse
import subprocess

# Create your views here.
def results(request):
    if request.method == 'POST':
        data = request.POST.get('name')
        print(data)
        context= {}
        system=request.POST.get('req_method', None )
        context['req_method']=system
        return render(request,'../templates/results.html',context)
    else:
        return render(request, 'results.html')
def index(request):
    data = request.POST
    print(data)
    return render(request, '../templates/index.html')

    
