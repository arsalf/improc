# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Image(models.Model):    
    image = models.ImageField(upload_to='apps/static/assets/images/originals')    
    task = models.CharField(max_length=100, null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.description

    def delete(self, *args, **kwargs):
        self.image.delete()
        super().delete(*args, **kwargs)
