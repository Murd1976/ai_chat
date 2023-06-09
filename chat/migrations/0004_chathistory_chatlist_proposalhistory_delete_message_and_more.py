# Generated by Django 4.1.7 on 2023-05-12 15:17

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0003_messagechain_job_title'),
    ]

    operations = [
        migrations.CreateModel(
            name='ChatHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True, db_index=True, verbose_name='Message time.')),
                ('user_question', models.TextField()),
                ('ai_answer', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='ChatList',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('chat_name', models.CharField(max_length=100)),
                ('created_at', models.DateTimeField(auto_now_add=True, db_index=True, verbose_name='Message time.')),
                ('owner', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to=settings.AUTH_USER_MODEL, verbose_name='Chat owner.')),
            ],
        ),
        migrations.CreateModel(
            name='ProposalHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True, db_index=True, verbose_name='Message time.')),
                ('user_question', models.TextField()),
                ('ai_answer', models.TextField()),
                ('message_chain', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='chat.messagechain')),
            ],
        ),
        migrations.DeleteModel(
            name='Message',
        ),
        migrations.AddField(
            model_name='chathistory',
            name='message',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='chat.chatlist'),
        ),
    ]
