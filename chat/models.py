from django.db import models
from django.utils import timezone
from django.contrib.auth.models import AbstractUser

version = 3.3

class AdvUser(AbstractUser):
    is_activated = models.BooleanField(default = True, db_index = True, verbose_name = 'Has been activated ?')
    send_messages = models.BooleanField(default = True, verbose_name = 'Send update messages ?')
    paid_account = models.BooleanField(default = False)
    
    class Meta(AbstractUser.Meta):
        pass

class ChatList(models.Model):
    owner = models.ForeignKey(AdvUser, verbose_name='Chat owner.', on_delete = models.DO_NOTHING)
    chat_name = models.CharField(max_length=100)
    
    created_at = models.DateTimeField(auto_now_add=True, db_index=True, verbose_name='Message time.')
    chat_type = models.CharField(max_length=30, default = 'none')
    company = models.CharField(max_length=100, default = 'none')
    subject = models.CharField(max_length=100, default = 'none')
    status = models.CharField(max_length=30, default = 'open')

    def __str__(self):
        return f"ChatName: {self.chat_name}"

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
        
class ProposalHistory(models.Model):
    message_chain = models.ForeignKey(MessageChain, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True, verbose_name='Message time.')
    user_question = models.TextField()
    ai_answer = models.TextField()

    def __str__(self):
        return f"ProposalHistory: {self.user_question} - {self.ai_answer}"
        
class ChatHistory(models.Model):
    message = models.ForeignKey(ChatList, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True, verbose_name='Message time.')
    user_question = models.TextField()
    ai_answer = models.TextField()

    def __str__(self):
        return f"ChatHistory: {self.user_question} - {self.ai_answer}"