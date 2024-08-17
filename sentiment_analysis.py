from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

#Select a model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#Create a pipeline
classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer)
results = classifier(["Today was a good day!",
                 "My Oura sleep score was pretty high last night"])

#Print the results
for result in results:
    print(result)
    
#Creating a sample input
X_train = ["Today was a good day!",
            "My Oura sleep score was pretty high last night"]

#Tokenize the input
batch = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
print(batch)

with torch.no_grad():
    #Get the model outputs
    outputs = model(**batch, labels=torch.tensor([1, 1]))
    print(f"outputs: {outputs}")
    #Get the predicted class
    predictions = F.softmax(outputs.logits, dim=1)
    print(f"predictions: {predictions}")
    labels = torch.argmax(predictions, dim=1)
    print(f"Labels: {labels}")
    labels = [model.config.id2label[label] for label in labels.tolist()]
    print(labels)
    
    




