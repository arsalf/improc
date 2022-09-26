from django import forms
from .models import Image

#DataFlair
class ImageCreate(forms.ModelForm):
    class Meta:
        model = Image
        fields = '__all__'