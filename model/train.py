import torch 
from transformers import AdamW
from dataset import model, data, label_map, JobTitleDataset, BertTokenizer, DataLoader

def get_dataloader():
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #Create dataset and dataloader
   dataset = JobTitleDataset(data, tokenizer)
   dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
   return (tokenizer, dataloader)


def train_model():
    #set up optimizer and training parameters
    optimizer = AdamW(model.parameters(), lr=1e-5)
    epochs = 3
    tokenizer, dataloader = get_dataloader()

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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = outputs.logits.argmax(dim=-1).item()

        title = list(label_map.keys())[list(label_map.values()).index(prediction)]
        return title

    print(generate_job_title("customer support"))

    #Save the model after training it into a state dictionary
    torch.save(model, 'bert_job_phrase_model.pth')

    #Save tokenizer 
    tokenizer.save_pretrained('./bert_tokenizer')


