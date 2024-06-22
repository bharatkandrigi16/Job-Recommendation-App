from transformers import BertTokenizer, BertForSequenceClassification
import torch 

#data pairs - assuming user is a college student looking for IT/Tech roles
data = [
    ("data science machine learning", "Data Scientist"),
    ("software development python", "Python Developer"),
    ("software engineering python", "Python Developer"),
    ("entry-level software engineering", "Software Engineer"),
    ("software C++", "C++ Developer"),
    ("engineering C++", "C++ Developer"),
    ("software engineering C++", "C++ Developer"),
    ("software Java", "Java Developer"),
    ("engineering Java", "Java Developer"),
    ("sofware engineering Java", "Java Developer"),
    ("data pipeline", "Data Engineer"),
    ("data analysis sql", "Data Analyst"),
    ("customer relations","Customer Success Manager"),
    ("customer support","Customer Service Representative"),
    ("testing","Quality Assurance Specialist"),
    ("test specialist","Quality Assurance Specialist") 
]

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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)

#Prepare dataset for training
from torch.utils.data import Dataset, DataLoader

class JobTitleDataset(Dataset):

    def __init__(self, data, tokenizer, max_length=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        keywords, title = self.data[idx]
        inputs = self.tokenizer.encode_plus(keywords, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        #labels = self.tokenizer.encode_plus(title, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')['input_ids']
        label = torch.tensor(label_map[title], dtype=torch.long)
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': label
        }
    
#Create dataset and dataloader
dataset = JobTitleDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
