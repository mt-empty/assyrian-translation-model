import torch
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

"""
This file provides examples on how to use the model
"""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("mt-empty/english-assyrian")
model = AutoModelForSeq2SeqLM.from_pretrained("mt-empty/english-assyrian")

translator = pipeline("translation", model=model, tokenizer=tokenizer, device=0)

print("Translation examples:")

print(
    "of the Eve of Preparation, of Passion",
    translator("of the Eve of Preparation, of Passion"),
)
print("of the Birth", translator("of the Birth"))
print("I'm working as a smith", translator("I'm working as a smith"))
print("I love you", translator("I love you"))
print("tomorrow, I will go to the shop", translator("tomorrow, I will go to the shop"))
print("how are you", translator("how are you"))
print("I'm good", translator("I'm good"))
print("I will see you on Monday", translator("I will see you on Monday"))
print("home", translator("home"))


metric = load_metric("sacrebleu")

print("\nComputing metric examples:")

predictions = ["ܚܙܰܝܢܰܢ ܐܰܬ݂ ܪܐ ܚܰܕ̱ܬ݂ ܐ، ܘܡܰ ܪܕܘܬ݂ ܐ ܚܕܰܬ݂ "]
references = [["ܡܳ ܝܰܬ݂ ، ܚܙܰܝܢܰܢ ܐܰܬ݂ ܪܐ ܚܰܕ̱ܬ݂ ܐ، ܘܡܰ ܪܕܘܬ݂ ܐ ܚܕܰܬ݂ ܐ"]]
print(metric.compute(predictions=predictions, references=references))
