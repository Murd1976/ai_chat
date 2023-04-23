from django.db import models
from django.utils import timezone
from django.contrib.auth.models import AbstractUser

version = 2.00

class AdvUser(AbstractUser):
    is_activated = models.BooleanField(default = True, db_index = True, verbose_name = 'Has been activated ?')
    send_messages = models.BooleanField(default = True, verbose_name = 'Send update messages ?')
    paid_account = models.BooleanField(default = False)
    
    class Meta(AbstractUser.Meta):
        pass

class Message(models.Model):
    owner = models.ForeignKey(AdvUser, verbose_name='Test owner.', on_delete = models.DO_NOTHING)
    job_description = models.CharField(max_length=50)
    proposal_letter = models.CharField(max_length=50)
    question_id = models.CharField(max_length=50)
    answer_id = models.CharField(max_length=50)
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True, db_index=True, verbose_name='Message time.')

    def __str__(self):
        return self.text

    @classmethod
    def get_all_messages(cls):
        return cls.objects.all()    
        
class MessageChain(models.Model):
    owner = models.ForeignKey(AdvUser, verbose_name='Chein owner.', on_delete = models.DO_NOTHING)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True, verbose_name='Message time.')
    job_title = models.TextField(default = "noname")
    job_description = models.TextField()
    proposal_cover_letter = models.TextField()
    chat_interview = models.TextField()

    def __str__(self):
        return f"MessageChain: {self.id}"
    
class QuestionAnswer(models.Model):
    message_chain = models.ForeignKey(MessageChain, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True, verbose_name='Message time.')
    question = models.TextField()
    answer = models.TextField()

    def __str__(self):
        return f"QuestionAnswer: {self.question} - {self.answer}"