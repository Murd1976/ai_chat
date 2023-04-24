import openai
import os
import django
from dotenv import load_dotenv
from .models import Message
from django.core.management.base import BaseCommand
from chat.models import MessageChain, QuestionAnswer



'''
class Command(BaseCommand):
    help = 'Train Chat GPT model on messages'
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    //os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
    //django.setup()

    def handle(self, *args, **options):
        # Authenticate OpenAI API
        openai.api_key = os.getenv('OPENAI_API_KEY')

        # Get all messages from the database
        messages = []
        for message_chain in MessageChain.objects.all():
            messages.extend([message_chain.job_description, message_chain.proposal_cover_letter, message_chain.chat_interview])
            for qa in QuestionAnswer.objects.filter(message_chain=message_chain):
                messages.extend([qa.question, qa.answer])

        # Add additional data from external sources, such as text files or APIs
        #with open('external_data.txt', 'r') as f:
        #    messages.extend(f.read().split('\n'))

        # Train the model on all messages
        model = openai.Model.create("text-davinci-002")
        model.train(training_data=messages, num_epochs=5)

        # Save the trained model
        model_id = 'my_chat_model_1'
        model.update(model_id=model_id)

        # Set the default model for future requests
        openai.api_model_id = model_id
        '''
        
# Authenticate OpenAI API
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
my_model_name = 'my_model_murd_001'

# Получаем ключ API
#api_key = "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"
#api_key = "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"
#openai.api_key = api_key

# Функция для получения ответа от ChatGPT
def get_response(model_id, query):
    # Формируем запрос к API
    print('---------------------------')
    print(query)
    print('---------------------------')
    response = openai.Completion.create(
        model= model_id,
        prompt=query,
        temperature=0.6,
        #max_tokens=400,
        top_p=1.0,
        stop=None,
        frequency_penalty=0.0,
        presence_penalty=0.0
        
    )
    print('+++++++++++++++++++++++++++++++++')
    print(response)
    print('+++++++++++++++++++++++++++++++++')
    # Получаем ответ
    reply = ""
    #reply = response["choices"][0]["text"] #response_json["choices"][0]["text"]
    for choice in response["choices"]:
        if choice["text"]:
            reply = choice["text"]
            break
    #print('The answer: ', reply)

    return reply
        
# Функция для получения ответа от ChatGPT
def get_proposal(query):
    # Формируем запрос к API
    model_id = my_model_name
    try:
        response = openai.Completion.create(
            model= model_id,
            prompt=query,
            temperature=0.6,
            #max_tokens=400,
            top_p=1.0,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0
           
        )
        print("Used model: " + model_id)
    except:
        response = openai.Completion.create(
            model= "text-davinci-003",
            prompt=query,
            temperature=0.6,
            max_tokens=700,
            top_p=1.0,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0
            
        )
        print("Used model: text-davinci-003")
        
    print('+++++++++++++++++++++++++++++++++')
    print(response)
    print('+++++++++++++++++++++++++++++++++')
    # Получаем ответ
    reply = ""
    #reply = response["choices"][0]["text"] #response_json["choices"][0]["text"]
    for choice in response["choices"]:
        if choice["text"]:
            reply = choice["text"]
            break
    
    #print('The answer: ', reply)
    query.extend(["\n Proposal: \n", reply])
    
    # Fine-tune a new model based on the existing ones
    model = openai.Model(
        id = model_id,
        model="text-davinci-003",
        fine_tune= model_id, #base_models,
        training_data= query,
        num_epochs=5
    )
    
    # Save the trained model
    #model.update(model_id=model_id)
   
    # Set the default model for future requests
    openai.api_model_id = model_id
    

    return reply
        
def train_chat():
    
    #openai.api_key = "YOUR_API_KEY"
    print("Start traning model!")
    
    # Get all messages from the database
    messages = []
    for message_chain in MessageChain.objects.all():
        
        messages.extend([" Job title: \n" + message_chain.job_title, '\n\n Job description: \n' + message_chain.job_description, '\n\n Proposal: \n' + message_chain.proposal_cover_letter])
        for qa in QuestionAnswer.objects.filter(message_chain=message_chain):
            messages.extend(["\n Question: \n" + qa.question, "\n Answer: \n" + qa.answer])

    # Add additional data from external sources, such as text files or APIs
    #with open('external_data.txt', 'r') as f:
    #    messages.extend(f.read().split('\n'))
    '''
    # Use multiple models as the starting point for training
    base_models = [
        "model-1-id",
        "model-2-id",
        "model-3-id"
    ]
    '''
    model_id = 'my_model_murd_001' #my_model_name
    # Initialize a new model instance and start fine-tuning it
    model = openai.Model(
        model="text-davinci-003",
        fine_tune=model_id,
        training_data=messages,
        num_epochs=5
    )

    
    # Save the trained model
    
    model.update(model_id=model_id)
           
    #status = model.retrieve(model_id)
    #print("Status: ", status)
    models = openai.Model.list()
    for mod in models["data"]:
        print(mod["id"])
   
    # Set the default model for future requests
    openai.api_model_id = model_id
    
    print(f"Stop traning model {model_id}!")