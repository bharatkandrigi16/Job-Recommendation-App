from django.shortcuts import render
from django.http import JsonResponse
from transformers import BertTokenizer, BertForSequenceClassification
import torch

#Load tokenizer
tokenizer = BertTokenizer.from_pretrained('./bert_tokenizer')

#Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.load_state_dict(torch.load('bert_job_phrase_model.pth'))
model.eval()

label_map = {
        "Data Scientist": 0,
        "Python Developer": 1,
        "Software Engineer": 2,
        "C++ Developer": 3,
        "Java Developer": 4,
        "Data Engineer": 5,
        "Data Analyst": 6,
        "Customer Success Manager": 7,
        "Customer Service Representative": 8,
        "Quality Assurance Specialist": 9,
    }

def home(req):
    return render(req, 'home.html', {})
    

def generate_job_title(keywords):
    model.eval()
    inputs = tokenizer.encode_plus(
        keywords,
        return_tensors = 'pt',
        max_length = 32, 
        truncation = True,
        padding = 'max_length'
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = outputs.logits.argmax(dim=-1).item()

    title = list(label_map.keys())[list(label_map.values()).index(prediction)]
    return title
