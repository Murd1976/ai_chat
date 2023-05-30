from django import forms
from django.contrib.auth import password_validation
from django.core.exceptions import ValidationError

from .models import AdvUser
from .apps import user_registered

version = 2.3

class UserForm(forms.Form):
    name = forms.CharField()
    age = forms.IntegerField()
    check1 = forms.ChoiceField(choices=((1, "English"), (2, "German"), (3, "French")), widget=forms.RadioSelect)

class ChangeUserInfoForm(forms.ModelForm):
    email = forms.EmailField(required= True, label= 'Email address')
    
    class Meta:
        model= AdvUser
        fields= ('username', 'email', 'first_name', 'last_name', 'send_messages')
        
class RegisterUserForm(forms.ModelForm):

    email = forms.EmailField(required= True, label= 'Email address')
    password1 = forms.CharField(label = 'Password', widget = forms.PasswordInput,
                               help_text = password_validation.password_validators_help_text_html(), required = True)
    password2 = forms.CharField(label = 'Password (again)', widget = forms.PasswordInput,
                               help_text = 'Enter the password again', required = True)

    def clean_password1(self):
        password1 = self.cleaned_data['password1']
        if password1:
            password_validation.validate_password(password1)
        return password1
    
    def clean(self):
        super().clean()
        if self.is_valid():
            pass2 = self.cleaned_data['password2']
            pass1 = self.cleaned_data['password1']
        
            if pass1 and pass2 and pass1 != pass2:
                errors = {'password2': ValidationError('Passwords do not match', code = 'password_mismatch')}
                raise ValidationError(errors)
        else: 
            errors = {'password1': ValidationError('Validation error', code = 'form error')}
            raise ValidationError(errors)
            
    def save(self, commit = True):
        user = super().save(commit = False)
        user.set_password(self.cleaned_data['password1'])
        user.is_active = False
        user.is_activated = False
        if commit:
            user.save()
        user_registered.send(RegisterUserForm, instance = user)
        return user
    
    class Meta:
        model= AdvUser
        fields= ('username', 'email', 'password1', 'password2', 'first_name', 'last_name', 'send_messages')
        
class ChatForm(forms.Form):
    ask_field = forms.CharField(widget= forms.Textarea(attrs={'class':'text_field', 'rows':'8', 'cols':'70'}), disabled = False, required=True)
    chat_field = forms.CharField(widget= forms.Textarea(attrs={'class':'text_field', 'rows':'15', 'cols':70}), disabled = False, required=False)
    
class JobForm(forms.Form):

    f_job_title = forms.CharField(label="Job tile", required=True, widget=forms.TextInput(attrs={'class':'str_input', 'placeholder': 'job title', 'size': 90}))
    f_job_description = forms.CharField(label="Job description", widget= forms.Textarea(attrs={'class':'text_field', 'rows':'8', 'cols':'80'}), disabled = False, required=True)
    f_propose = forms.CharField(label="Proposal", widget= forms.Textarea(attrs={'class':'text_field', 'rows':'15', 'cols':'80'}), disabled = False, required=False)
    f_model = forms.ChoiceField(label="Select an AI model:", initial=0, required= True, choices=((0, "text-davinci-003"), (1, "gpt-3.5-turbo")))
    
class EditJobForm(forms.Form):

    f_content = forms.CharField(label="Proposal", widget= forms.Textarea(attrs={'class':'text_field', 'rows':'15', 'cols':'80'}), disabled = False, required=True)
    f_feedback = forms.CharField(label="FeedBack", widget= forms.Textarea(attrs={'class':'text_field', 'rows':'8', 'cols':'80'}), disabled = False, required=False)
    f_model = forms.ChoiceField(label="Select an AI model:", initial=0, required= True, choices=((0, "text-davinci-003"), (1, "gpt-3.5-turbo")))
    
class OutTextForm(forms.Form):
    f_content = forms.CharField(label="Status", widget= forms.Textarea(attrs={'class':'text_field', 'rows':'15', 'cols':'80'}), disabled = False, required=True)
    
class DbLoadForm(forms.Form):
    db_name = forms.CharField(required=True)