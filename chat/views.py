from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.template import TemplateDoesNotExist
from django.template.loader import get_template
from django.contrib import messages
from django.contrib.messages.views import SuccessMessageMixin
from django.contrib.auth import logout
from django.contrib.auth.views import LoginView, LogoutView, PasswordChangeView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.views.generic.edit import UpdateView, CreateView, DeleteView
from django.views.generic.base import TemplateView

from django.core.signing import BadSignature
from django.utils.encoding import smart_str
from django.shortcuts import get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound, HttpResponseBadRequest, Http404

from .forms import *
from .models import *
from .utilities import load_from_xls
from .ai_train import *

import pandas as pd
import openai
import requests
from dotenv import load_dotenv
import os

class BBLoginView(LoginView):
    template_name = 'chat/chat_login.html'
    
class BBLogoutView(LoginRequiredMixin, LogoutView):
    template_name = 'chat/chat_logout.html'

class RegisterDoneView(TemplateView):
    template_name = 'chat/chat_register_done.html'
    
class RegisterUserView(CreateView):
    model = AdvUser
    template_name = 'chat/chat_register_user.html'
    form_class = RegisterUserForm
    success_url = reverse_lazy('chat:my_register_done')

class ChangeUserInfoView(SuccessMessageMixin, LoginRequiredMixin, UpdateView):
    model = AdvUser
    template_name = 'chat/chat_change_user_info.html'
    form_class = ChangeUserInfoForm
    success_url = reverse_lazy('chat:my_profile')
    success_message = 'User data changed'
    
    def setup(self, request, *args, **kwargs):
        self.user_id = request.user.pk
        return super().setup(request, *args, **kwargs)
    
    def get_object(self, queryset=None):
        if not queryset:
            queryset = self.get_queryset()
        return get_object_or_404(queryset, pk=self.user_id)
    
class BBPasswordChangeView(SuccessMessageMixin, LoginRequiredMixin, PasswordChangeView):
    template_name = 'chat/chat_password_change.html'
    success_url = reverse_lazy('chat:my_profile')
    success_message = 'User password changed'
    
class DeleteUserView(LoginRequiredMixin, DeleteView):
    model = AdvUser
    template_name = 'chat/chat_delete_user.html'
    success_url = reverse_lazy('chat:index')
    
    def setup(self, request, *args, **kwargs):
        self.user_id = request.user.pk
        return super().setup(request, *args, **kwargs)
    
    def post(self, request, *args, **kwargs):
        logout(request)
        messages.add_message(request, messages.SUCCESS, 'User deleted')
        return super().post(request, *args, **kwargs)
    
    def get_object(self, queryset = None):
        if not queryset:
            queryset = self.get_queryset()
        return get_object_or_404(queryset, pk = self.user_id)
    
def index(request):   
    content = 'The homepage of AI Chat.'
    context = {"content": content}
    return render(request, "index.html", context)

@login_required
def user_profile(request):
    return render(request, 'chat/chat_profile.html')

def user_activate(request, sign):
    try:
        username = signer.unsign(sign)
    except BadSignature:
        return render(request, 'chat/chat_bad_signature.html')
    
    user = get_object_or_404(AdvUser, username = username)
    if user.is_activated:
        template = 'chat/chat_user_is_activated.html'
    else:
        template = 'chat/chat_activation_done.html'
        user.is_active = True
        user.is_activated = True
        user.save()
    return render(request, template)# Create your views here.


@login_required
def load_base_page(request):

    f_name = "chat/upwork_list.xlsx"
    if request.method == "POST":
        userform = DbLoadForm(request.POST or None)
        if userform.is_valid():
            f_name = userform.cleaned_data["db_name"]
            load_from_xls(f_name, request.user)
    parts = DbLoadForm(initial= {"db_name":f_name})                
    template = 'chat/chat_load_base.html'
    chain_list = MessageChain.objects.filter(owner=request.user)
#    tests_list = AllBackTests.objects.all()
#    tests_list.delete()
#    tests_list = AllBackTests.objects.all()
    context = {"form": parts, "tests_log": chain_list, "user_name":request.user}
    return render(request, template, context)


# Получаем ключ API
#api_key = "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"

# Инициализируем OpenAI
#openai = OpenAI(api_key)

#openai.api_key = api_key


# Функция для обработки формы диалога
def gpt_response(request):
    # Получаем ответ от ChatGPT
    query = "Extract the name and mailing address from this email:\n\nDear Kelly,\n\nIt was great to talk to you at the seminar. I thought Jane's talk was quite good.\n\nThank you for the book. Here's my address 2111 Ash Lane, Crestview CA 92002\n\nBest,\n\nMaya\n\nName:" 
    #"Name all the presidents of the USA"
    response = get_response("text-davinci-003", query)
    

    # Отправляем ответ пользователю
    context = {"question": query, "gpt_response": response, "user_name":request.user}
    return render(request, "chat/chat_gpt_response.html", context)
    
    
def train_page(request):
    #train_chat()
    
    context = {"train_res": "Training of model complited!"}
    return render(request, "chat/chat_train_model.html", context)
    
def train_status_page(request):
    response = "No treined models"
    #response = train_status("ft-7QqiAyTvtmME8WHn6ZKAQa07")
    parts = OutTextForm(initial= {"f_content":response})
    context = {"form": parts}
    return render(request, "chat/chat_train_status.html", context)

def other_page(request, page):
    try:
        template = get_template('chat/' + page + '.html')
    except TemplateDoesNotExist:
        raise Http404
    return HttpResponse(template.render(request = request))

def chat_home(request):
    data = {"header": "Main window", "message": "Welcome to Chat!"}
    return render(request, "chat/chat_home.html", context=data)
    
# delete record
def delete_record_new(request, id):
    try:
        test = MessageChain.objects.get(id=id)
        test.delete()
        return redirect('chat:my_new_proposal')
    except MessageChain.DoesNotExist:
        return HttpResponseNotFound("<h2>Record not found</h2>")

# delete record
def delete_record_edit(request, id):
    try:
        test = MessageChain.objects.get(id=id)
        test.delete()
        return redirect('chat:my_edit_proposal')
    except MessageChain.DoesNotExist:
        return HttpResponseNotFound("<h2>Record not found</h2>")
        

def chat_page(request):
    template = 'chat/chat_gpt_response.html'
        
    if request.method == "POST":
        #text_buf = "Amswer: "
        userform = ChatForm(request.POST or None)
        #text_buf += dict(strategies_value)[str(userform.data.get("f_strategies"))]
                
        if userform.is_valid():
            query = userform.cleaned_data["ask_field"]
            
            #data_bufer = DataBufer.objects.filter(name=request.user)
            #data_bufer.delete()
            
            #data_bufer = DataBufer(name=request.user, user_strategy_choise=strategy_choise)
            #data_bufer.save()
            
            messages = []
            #message_chain = MessageChain.objects.filter(owner = request.user).last()
            #messages.extend([" Job title: \n" + message_chain.job_title, '\n\n Job description: \n' + message_chain.job_description, '\n\n Proposal: \n' + message_chain.proposal_cover_letter, "\n\n Create new Proposal for this job."])
            #messages.extend([" Job title: \n" + message_chain.job_title, '\n\n Job description: \n' + message_chain.job_description, '\n\n Proposal: \n', "\n\n Create Proposal for this job."])
            
            #messages.extend(["\n Question: \n", query])
            #messages.extend([message_chain.job_title])
            #print(messages)
            '''
            res = QuestionAnswer.objects.filter(message_chain=message_chain)
            
            for qa in res:
                if not (pd.isna(qa.question)):
                    messages.extend(["\n Question: \n", qa.question, "\n Answer: \n", qa.answer])
            '''
                                    
            response = get_response("text-davinci-003", query)
            #print(response)
            '''
            # сохраняем ответ в базу данных
            QuestionAnswer.objects.create(
                message_chain = MessageChain.objects.last(),
                question = query,
                answer = response.strip()
            )
            '''
            
            #print(response.strip())
            messages.extend(["\n Answer: \n", response.strip()])
            st = ""
            for buf in messages:
                st += buf
            parts = ChatForm(initial= {"chat_field":st, "ask_field":query})
            #parts.fields['f_reports'].choices = reports_list
            
            context = {"form": parts}
            return render(request, template, context)
        else:
            return HttpResponse("Something was wrong...")

    parts = ChatForm()
    context = {"form": parts}
    return render(request, template, context)
  
def new_proposal_page(request, id = 0):
    template = 'chat/chat_new_proposal.html'
    jobs = MessageChain.objects.all()
    models = [(0, 'text-davinci-003'), (1, 'gpt-3.5-turbo')]
    if request.method == "POST":
        
        userform = JobForm(request.POST or None)
                
        if userform.is_valid():
            sys_msg = " You are an assistant who creates a proposal cover letter for new job offers."
            query_text = " Create new proposal cover letter for job title and job description below.:\n\n" 
            
            
            j_title = userform.cleaned_data["f_job_title"] 
            j_description = userform.cleaned_data["f_job_description"]
            curr_model = dict(models)[int(userform.cleaned_data["f_model"])]
            #print(f"\n Current model: {curr_model} \n")
            #data_bufer = DataBufer.objects.filter(name=request.user)
            #data_bufer.delete()
            
            #data_bufer = DataBufer(name=request.user, user_strategy_choise=strategy_choise)
            #data_bufer.save()
           
            messages = []
            count = 0
            for message_chain in MessageChain.objects.all():
                #if count > 1: break
                messages.extend([ " Job title: \n" + message_chain.job_title + '\n\n Job description: \n' + message_chain.job_description + '\n\n Proposal: \n' + message_chain.proposal_cover_letter])
                count += 1
            #messages.extend([message_chain.job_title])
            #print(messages)
            
            #print(messages)
            #print()
            #query.extend([messages, " Job title: \n" + j_title +  '\n\n Job description: \n' + j_description +  '\n\n Proposal: \n'])
            #query = f"Create new proposal cover letter for job title and job description below\n\n {messages[7]}\n\n Proposal:\n"
            
            #curr_job_chain = MessageChain.objects.filter(owner = request.user).last()
            curr_job_chain = MessageChain.objects.last()
            query = f"{query_text}  Job title: {j_title}\n\n Job description: {j_description}\n\n Example of proposal: {curr_job_chain.proposal_cover_letter}\n"
            
            if (curr_model == 'text-davinci-003'):
                    response = get_proposal(query)
            if (curr_model == 'gpt-3.5-turbo'):
                    response = get_proposal_turbo(query, sys_msg)
            #print(response)
            
            # сохраняем ответ в базу данных
            MessageChain.objects.create(
                owner = request.user,
                job_title = j_title,
                job_description = j_description,
                #proposal_cover_letter = response.strip(),
                proposal_cover_letter = response,
                chat_interview = ""
            )
            
            # создаем новый чат для нового предложения и сохраняем в базу данных
            ProposalHistory.objects.create(
                message_chain = MessageChain.objects.last(),
                user_question = query,
                #ai_answer = response.strip()
                ai_answer = response
            )
                                    
            parts = JobForm(initial= {"f_job_title":j_title, "f_job_description":j_description, "f_propose":response})
                        
            context = {"form": parts, "jobs": jobs}
            return render(request, template, context)
        else:
            return HttpResponse("Something was wrong...")
    
    if (id > 0): 
        curr_job_chain = MessageChain.objects.get(id=id)
    else:
        curr_job_chain = MessageChain.objects.last()
            
    if (curr_job_chain == None):
        parts = JobForm()
    else:
        
        parts = JobForm(initial= {"f_job_title":curr_job_chain.job_title, "f_job_description":curr_job_chain.job_description, "f_propose":curr_job_chain.proposal_cover_letter})
        
    context = {"form": parts, "jobs": jobs}
    return render(request, template, context)
    
def edit_proposal_page(request, id = 0):
    template = 'chat/chat_edit_proposal.html'
    jobs = MessageChain.objects.all()
    models = [(0, 'text-davinci-003'), (1, 'gpt-3.5-turbo')]
           
    if request.method == "POST":
        
        userform = EditJobForm(request.POST or None)
        
        if userform.is_valid():
            #query = [] 
            sys_msg = "You are an assistant who, based on the user's comments, edits the current cover letter for new job offers."
            if(id == 0): 
                template = 'chat/chat_msg_page.html'
                context = {"my_msg":"First you need to select an entry!"}
                #return HttpResponse("First you need to select an entry!")
                return render(request, template, context)
            
            j_feedback = userform.cleaned_data["f_feedback"]
            #curr_job_chain = MessageChain.objects.filter(owner = request.user).get(id = id)
            curr_job_chain = MessageChain.objects.get(id = id)
            curr_model = dict(models)[int(userform.cleaned_data["f_model"])]
            
            #data_bufer = DataBufer.objects.filter(name=request.user)
            #data_bufer.delete()
            
            #data_bufer = DataBufer(name=request.user, user_strategy_choise=strategy_choise)
            #data_bufer.save()
            
            #messages = ['Feedback: ' + j_feedback]
            messages = []
            #text = 'Examples: '
            count = 0
            for message_chain in MessageChain.objects.all():
                if count > 10: break
                messages.extend([ " Job title: \n" + message_chain.job_title + '\n\n Job description: \n' + message_chain.job_description + '\n\n Proposal: \n' + message_chain.proposal_cover_letter])
                count += 1
            
            #messages.extend([message_chain.job_title])
            #print(messages)
            
            #print(messages)
            #print()
            #query.extend([ " Job title: \n" + j_title,  '\n\n Job description: \n' + j_description, "\n\n Create new proposal for this job"])
            #messages.extend([" Job title: \n" + curr_job_chain.job_title + '\n\n Job description: \n' + curr_job_chain.job_description + 
            #                    '\n\n Proposal: \n' + curr_job_chain.proposal_cover_letter])
            example = " Job title: \n" + curr_job_chain.job_title + '\n\n Job description: \n' + curr_job_chain.job_description + '\n\n Proposal: \n' + curr_job_chain.proposal_cover_letter
            query = f"{j_feedback}: \n\n {example}\n\n New Proposal:\n"
            #print(query)
            #query.extend(messages)
            #response = get_proposal(query)
            if (curr_model == 'text-davinci-003'):
                    response = get_proposal(query)
            if (curr_model == 'gpt-3.5-turbo'):
                    response = get_proposal_turbo(query, sys_msg)
            
            curr_job_chain.proposal_cover_letter = response.strip()
            curr_job_chain.save(update_fields=["proposal_cover_letter"])
            
            # создаем новую запись в чате для текущего предложения
            ProposalHistory.objects.create(
                message_chain = curr_job_chain,
                user_question = query,
                ai_answer = response.strip()
            )
            #print(response)
            '''
            # сохраняем ответ в базу данных
            MessageChain.objects.create(
                owner = request.user,
                job_title = j_title,
                job_description = j_description,
                proposal_cover_letter = response.strip(),
                chat_interview = ""
            )
            '''                        
            
            messages = []
            res = ProposalHistory.objects.filter(message_chain=curr_job_chain)
        
            if(len(res) == 0):
                messages.extend(curr_job_chain.proposal_cover_letter)
            else:
                for qa in res:
                    if not (pd.isna(qa.user_question)):
                        messages.extend(["\n ______________________________________________________ \n", qa.user_question, "\n ______________________________________________________ \n", qa.ai_answer])
                
            st = ""
            for buf in messages:
                st += buf
            parts = EditJobForm(initial= {"f_content":st})

            
            #parts = EditJobForm(initial= {"f_content":response.strip()})
                        
            context = {"form": parts, "jobs": jobs}
            return render(request, template, context)
        else:
            return HttpResponse("Something was wrong...")
            
    if (id > 0): 
        curr_job_chain = MessageChain.objects.get(id=id)
    else:
        curr_job_chain = MessageChain.objects.last()
      
    if (curr_job_chain == None):
        parts = JobForm()
    else:
        #query = []
        #query.extend([curr_job_chain.proposal_cover_letter])
        messages = []
        res = ProposalHistory.objects.filter(message_chain=curr_job_chain)
        
        if(len(res) == 0):
            messages.extend(curr_job_chain.proposal_cover_letter)
        else:
            for qa in res:
                if not (pd.isna(qa.user_question)):
                    messages.extend(["\n ______________________________________________________ \n", qa.user_question, "\n ______________________________________________________ \n", qa.ai_answer])
        #parts = JobForm(initial= {"f_job_title":curr_job_chain.job_title, "f_job_description":curr_job_chain.job_description, "f_propose":messages})
        
        st = ""
        for buf in messages:
            st += buf
        parts = EditJobForm(initial= {"f_content":st})
        
    context = {"form": parts, "jobs": jobs}
    return render(request, template, context)
    
def m304(request):
    return HttpResponseNotModified()
 
def m400(request):
    return HttpResponseBadRequest("<h2>Bad Request</h2>")
 
def m403(request):
    return HttpResponseForbidden("<h2>Forbidden</h2>")
 
def m404(request):
    return HttpResponseNotFound("<h2>Not Found</h2>")
 
def m405(request):
    return HttpResponseNotAllowed("<h2>Method is not allowed</h2>")
 
def m410(request):
    return HttpResponseGone("<h2>Content is no longer here</h2>")
 
def m500(request):
    return HttpResponseServerError("<h2>Something is wrong</h2>")