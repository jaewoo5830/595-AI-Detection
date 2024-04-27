from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# create the datasets for BERT Training
dataset = load_dataset('csv', data_files={
    'train': 'train.csv',
    'validation': 'dev.csv',
    'test': 'test.csv'
})

# Initialize the tokenizers
DistilBert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
RoBERTa_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Create Tokenize Functions
def DistilBERT_tokenize_function(data):
    return DistilBert_tokenizer(data['text'], padding="max_length", truncation=True)

def RoBERTa_tokenize_function(data):
    return RoBERTa_tokenizer(data['text'], padding="max_length", truncation=True, max_length=512)

# Tokenize the Dataset
DistilBert_tokenized_datasets = dataset.map(DistilBERT_tokenize_function, batched=True)
RoBERTa_tokenized_datasets = dataset.map(RoBERTa_tokenize_function, batched=True)

# Load Models
DistilBert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
RoBERTa = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Create Training Args
DistilBert_training_args = TrainingArguments(
    output_dir='./results_distill',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_distill',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,  # load the best model at the end of training
    metric_for_best_model="f1",  # use accuracy to identify the best model
)

RoBERTa_training_args = TrainingArguments(
    output_dir='./results_roberta',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_roberta',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# Create compute metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
}

# Create trainers for each model
DistilBert_trainer = Trainer(
    model=DistilBert,
    args=DistilBert_training_args,
    train_dataset=DistilBert_tokenized_datasets['train'],
    eval_dataset=DistilBert_tokenized_datasets['validation'],
    compute_metrics=compute_metrics
)

RoBERTa_trainer = Trainer(
    model=RoBERTa,
    args=RoBERTa_training_args,
    train_dataset=RoBERTa_tokenized_datasets['train'],
    eval_dataset=RoBERTa_tokenized_datasets['validation'],
    compute_metrics=compute_metrics
)

# Train and evaluate the models
print("Training DistilBert...")
DistilBert_trainer.train()

print("Evaluating DistilBert...")
DistilBert_trainer_test_results = DistilBert_trainer.evaluate(DistilBert_tokenized_datasets['test'])
print(DistilBert_trainer_test_results)

print("Training RoBERTa...")
RoBERTa_trainer.train()

print("Evaluating RoBERTa...")
RoBERTa_trainer_test_results = RoBERTa_trainer.evaluate(RoBERTa_tokenized_datasets['test'])
print(RoBERTa_trainer_test_results)

# Save the models
DistilBert.save_pretrained('DistillBERT')
DistilBert_tokenizer.save_pretrained('DistillBERTTokenizer')

RoBERTa.save_pretrained('RoBERTa')
RoBERTa_tokenizer.save_pretrained('RoBERTaTokenizer')
