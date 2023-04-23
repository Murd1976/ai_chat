import pandas as pd
import numpy as np
from django.template.loader import render_to_string
from django.core.signing import Signer

from ai_chat.settings import ALLOWED_HOSTS

signer = Signer()

def send_activation_notification(user):
    if ALLOWED_HOSTS:
        host = 'http://' + ALLOWED_HOSTS[0]
    else:
        host = 'http://localhost:8000'
    context = {'user':user, 'host':host, 'sign':signer.sign(user.username)}
    subject = render_to_string('chat/email/chat_activation_letter_subject.txt', context)
    body_text = render_to_string('chat/email/chat_activation_letter_body.txt', context)
    user.email_user(subject.replace('\n', ''), body_text)

def load_from_xls(f_name, user):
    from .models import MessageChain, QuestionAnswer
    
    MessageChain.objects.all().delete()
    print(f_name)
    db = pd.read_excel(f_name)
    db = db.replace({np.NaN:None})
    for col_name, data in db.iterrows():
        # сохраняем в базу данных
        if (data['Chat Interview'] != None) : buf = data['Chat Interview']
        else: buf = ""
        message_ch = MessageChain.objects.create(
            owner = user,
            job_title = data["JOB TITLE"],
            job_description = data['JOB DESCRIPTION'],
            proposal_cover_letter = data['PROPOSAL COVER LETTER'],
            chat_interview = buf
        )
        message_ch.save()
        '''
        print()
        print(message_ch.job_title)
        print('---------------')
        print(message_ch.chat_interview)
        print()
        '''
        for i in range(0, 5):
            q_str = "Question " + str(i+1)
            a_str = "Answer " + str(i+1)
            if(data[q_str] != None):
                #print(data[q_str], ' : ')
                qa = QuestionAnswer.objects.create(
                    message_chain = message_ch,
                    question = data[q_str],
                    answer = data[a_str]
                )
                qa.save()
