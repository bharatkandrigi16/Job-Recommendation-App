import torch 
from transformers import AdamW
from dataset import model, dataset, dataloader, tokenizer
from django.http import JsonResponse

#set up optimizer and training parameters

optimizer = AdamW(model.parameters(), lr=1e-5)
epochs = 3

#training loop
model.train()
for epoch in range(epochs):
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


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
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=32)

    title = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return JsonResponse({'job_title': title})


print(generate_job_title("data science"))

#Save the model after training it into a state dictionary
torch.save(model, 'bert_job_phrase_model.pth')

#Save tokenizer 
tokenizer.save_pretrained('./bert_tokenizer/')


