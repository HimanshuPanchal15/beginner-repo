# Generated by Django 4.1.2 on 2022-10-16 20:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app1', '0010_remove_profile_topics'),
    ]

    operations = [
        migrations.AddField(
            model_name='profile',
            name='avatar',
            field=models.ImageField(blank=True, null=True, upload_to='static/assets/avatars/', verbose_name='profile picture'),
        ),
    ]
