# Generated by Django 4.0.1 on 2022-09-25 11:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0002_remove_image_description'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='task',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
