import pandas as pd
import torch
from datasets import load_dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import Trainer, TrainingArguments

data = load_dataset("MartinThoma/wili_2018")
df = pd.DataFrame(data['train'])

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=len(df['label'].unique()))

def preprocess_function(examples):
    return tokenizer(
        examples['sentence'], 
        truncation=True, 
        padding='max_length', 
        max_length=256
    )

tokenized_datasets = data['train'].map(preprocess_function, batched=True)
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test_split['train'],
    eval_dataset=train_test_split['test'],
)

trainer.train()

def predict_language(text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding='max_length', 
        max_length=256
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return df['label'].unique()[predicted_class_id]

new_texts = [
    "ब्रेकिंग न्यूज़, वीडियो, ऑडियो और फ़ीचर",
]

for text in new_texts:
    lang_prediction = predict_language(text)
    print(f'Text: "{text}" is predicted to be in language: {lang_prediction}')
