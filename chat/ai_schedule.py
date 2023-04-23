from datetime import timedelta
from django.utils import timezone
from django_apscheduler.jobstores import DjangoJobStore
from django_apscheduler.models import DjangoJobExecution
from apscheduler.schedulers.background import BackgroundScheduler
from .models import Message

from .ai_train import train_chat
'''
def train_chat():
    messages = Message.objects.all()
    train_chat_gpt_model(messages)
'''
scheduler = BackgroundScheduler()
scheduler.add_jobstore(DjangoJobStore(), "default")

train_chat_interval_minutes = 60

scheduler.add_job(
    train_chat,
    "interval",
    minutes=train_chat_interval_minutes,
    id="train_chat",
    replace_existing=True,
)

def clean_old_chat():
    max_age_days = 30
    old_messages = Message.objects.filter(timestamp__lt=timezone.now()-timedelta(days=max_age_days))
    old_messages.delete()

scheduler = BackgroundScheduler()
scheduler.add_jobstore(DjangoJobStore(), "default")

clean_old_chat_interval_minutes = 24*60

scheduler.add_job(
    clean_old_messages_job,
    "interval",
    minutes=clean_old_chat_interval_minutes,
    id="clean_old_chat",
    replace_existing=True,
)

scheduler.start()
print(f"Scheduler started at {timezone.now()}.")