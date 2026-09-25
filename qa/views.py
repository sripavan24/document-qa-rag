from django.http import HttpResponse


def home(request):
    return HttpResponse("Django app is ready for the Basic Document Q&A Bot.")
