from django.shortcuts import render
from django.http import JsonResponse
from transformers import BertTokenizer, BertForSequenceClassification
import torch

#Load tokenizer
tokenizer = BertTokenizer.from_pretrained('./bert_tokenizer/')

#Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.load_state_dict(torch.load('bert_job_phrase_model.pth'))
model.eval()


def generate_job_title(request):
    keywords = request.GET.get('keywords', '')
    if not keywords:
        return JsonResponse({'error': 'No keywords provided'}, status=400)

    inputs = tokenizer.encode_plus(
        keywords,
        return_tensors='pt',
        max_length=32,
        truncation=True,
        padding='max_length'
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = outputs.logits.argmax(dim=-1)
    
        title = tokenizer.decode(predictions[0], skip_special_tokens=True)
        return JsonResponse({'job_title': title})
