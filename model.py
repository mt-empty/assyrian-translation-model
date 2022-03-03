import csv
import json
from datetime import datetime

import datasets
import numpy as np
import pandas
import sentencepiece as spm
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    MarianTokenizer,
    get_scheduler,
    pipeline,
)

# Constants

EPOCHS = 50
LEARNING_RATE = 5e-4
BATCH_SIZE = 180
ENGLISH_VOCAB_SIZE = 7861
ASSYRIAN_VOCAB_SIZE = int(ENGLISH_VOCAB_SIZE * 0.8)  # Calculated using calc_vocab.py
SHARED_VOCAB_SIZE = (
    ENGLISH_VOCAB_SIZE + ASSYRIAN_VOCAB_SIZE
)  # Calculated using calc_vocab.py
BASE_MODEL = "Helsinki-NLP/opus-mt-en-ar"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
metric = datasets.load_metric("sacrebleu")

# Step 1: Reading the the data, read into 2 files english and assyrian/syriac.

english_file = "en-as.en"
assyrian_file = "en-as.as"

# add pericopes data
df = pandas.read_csv("dataset/pericopes_data.csv", header=None).drop_duplicates(
    keep="first"
)
with open(english_file, "w", encoding="utf-8") as en_f, open(
    assyrian_file, "w", encoding="utf-8"
) as as_f:
    for enline, asline in zip(df[0], df[1]):
        en_f.write(str(enline) + "\n")
        as_f.write(str(asline) + "\n")


# add bible data
with open(english_file, "a", encoding="utf-8") as en_f, open(
    assyrian_file, "a", encoding="utf-8"
) as as_f, open("dataset/bible_data.csv", encoding="utf-8") as bd:
    csv_file = csv.reader(bd)
    for line in csv_file:
        en_f.write(str(line[0]) + "\n")
        as_f.write(str(line[1]) + "\n")


# Step 2: Create the dataset
mydataset = datasets.load_dataset(
    "config.py",
    data_dir="./",
    data_files=[english_file, assyrian_file],
    lang1="en",
    lang2="as",
)


# Step 3: Train a tokenizer
# We use SentencePiece to generates files needed for tokenization,
# these files contain the vocabulary and how the tokenization was done
spen = spm.SentencePieceTrainer.train(
    f"--input={english_file} --model_prefix=source --vocab_size={ENGLISH_VOCAB_SIZE} --unk_piece=<unk>"
)
spas = spm.SentencePieceTrainer.train(
    f"--input={assyrian_file} --model_prefix=target --vocab_size={ASSYRIAN_VOCAB_SIZE}--unk_piece=<unk>"
)
spm.SentencePieceTrainer.train(
    f"--input={english_file},{assyrian_file} --model_prefix=shared --vocab_size={SHARED_VOCAB_SIZE} --unk_piece=<unk>"
)

# Places the generated shared vocabulary of the two languages into a json file

shared_vocab = "shared_vocab.json"
with open("shared.vocab", encoding="utf-8") as vf:
    vocab_dict = {}
    for i, line in enumerate(vf):
        currentLine = line.split()
        vocab_dict[currentLine[0]] = i
    with open(shared_vocab, "w", encoding="utf-8") as shared_file:
        # it complains if we don't add <pad> character to shared_vocab
        vocab_dict["<pad>"] = SHARED_VOCAB_SIZE
        json.dump(vocab_dict, shared_file, sort_keys=False)

# we use the MarianTokenizer as our tokenizer
tokenizer = MarianTokenizer(f"./{shared_vocab}", "./source.model", "./target.model")


# example of translation before training, it only goes one way english -> arabic
translator = pipeline("translation", model=BASE_MODEL, device=0)

# lets try translating english -> arabic
print(translator("time"))

print(translator("home"))

print(translator("in the Sixth Hour of the Great Eve of Preparation"))

# lets try arabic -> english, it will most likely fail because the model only translates one way :(
print(translator("وقت"))
# so, to translate we need to use "Helsinki-NLP/opus-mt-ar-en" instead

# lets try Assyrian/Syriac -> english, again this will fail, same problem
print(translator("ܕܚܕܒܫܒܐ"))

# lets encode an english sentence to see if our tokenizer is working
print(tokenizer.encode("hello world"))


# splitting the dataset into training and validation
split_datasets = mydataset["train"].train_test_split(
    test_size=0.2, train_size=0.8, seed=20
)

# remove test set and add it to the validation set
split_datasets["validation"] = split_datasets.pop("test")

# Step 4: Training the model

# max amount of characters to be tokenized
max_input_length = 128
max_target_length = 128


# this will preprocess every token
def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["as"] for ex in examples["translation"]]

    # input language
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Set up the tokenizer for targets, switching to assyrian tokenizer
    with tokenizer.as_target_tokenizer():
        # target language language
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# tokenizing the dataset
tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)

# This shows an examples of our tokenized sentences
print(tokenized_datasets["train"][111]["labels"])
print(tokenized_datasets["train"][111]["input_ids"])

# The frist tokenized sentence is decoded into Assyrian
print(tokenizer.decode(tokenized_datasets["train"][111]["labels"]))

# The second tokenized sentence is decoded into English
print(tokenizer.decode(tokenized_datasets["train"][111]["input_ids"]))


# Generates a sequence to sequence English to Assyrian model from Helsinki as per the model_checkpoint variable
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

# Creates a data_collator object that can be used in the dataloader so that the dataset can be read by the model
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)

# TODO, this is for evaluation method
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=BATCH_SIZE, collate_fn=data_collator
)

for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

# use Adam optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

num_training_steps = EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

model.to(device)
print(f" the device is {device}")


# Training using Accelerator, for distributed GPU training


def postprocess(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels


accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

progress_bar = tqdm(range(num_training_steps))

for epoch in range(EPOCHS):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Step 5: Evaluation
    model.eval()
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
            )
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(generated_tokens)
        labels_gathered = accelerator.gather(labels)

        decoded_preds, decoded_labels = postprocess(
            predictions_gathered, labels_gathered
        )
        # print(f"pred: {decoded_preds}")
        # print(f"label: {decoded_labels}")
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    # if it fails due to empty/None refrence refer to https://github.com/mjpost/sacrebleu/pull/175
    results = metric.compute()
    print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")

    # Save model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        f"model_en-as_{epoch}_epochs_{datetime.today().strftime('%Y-%m-%d_%H-%M')}",
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(
            f"tokenizer_en-as_{epoch}_epochs_{datetime.today().strftime('%Y-%m-%d_%H-%M')}"
        )


# Training without Distributed GPU

# progress_bar = tqdm(range(num_training_steps))

# model.train()
# for epoch in range(EPOCHS):
#     for batch in train_dataloader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()

#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)

# save the model locally
# model.save_pretrained(
#     f"en-as_{EPOCHS}_epochs_{datetime.today().strftime('%Y-%m-%d_%H-%M')}"
# )


# Step 5: Evaluation

# now onto the fun part
# translation, get sample text from here:
# https://syriac.school/pluginfile.php/158/mod_resource/content/11/Text_samples.pdf


# sacreBleu, expects input to be at least of size 4 grams, otherwise it will output a score of 0

# metric = datasets.load_metric("sacrebleu")
# model.eval()
# for batch in eval_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**batch)

#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     metric.add_batch(predictions=predictions, references=batch["labels"])

# metric.compute()


# custom examples
print("Testing metric")
predictions = ["ܚܙܰܝܢܰܢ ܐܰܬ݂ ܪܐ ܚܰܕ̱ܬ݂ ܐ، ܘܡܰ ܪܕܘܬ݂ ܐ ܚܕܰܬ݂ "]
references = [["ܡܳ ܝܰܬ݂ ، ܚܙܰܝܢܰܢ ܐܰܬ݂ ܪܐ ܚܰܕ̱ܬ݂ ܐ، ܘܡܰ ܪܕܘܬ݂ ܐ ܚܕܰܬ݂ ܐ"]]
print(f"prediction: {predictions}")
print(f"reference: {references}")
print(metric.compute(predictions=predictions, references=references))

print("Testing translation after training")
translator = pipeline("translation", model=model, tokenizer=tokenizer, device=0)
print(translator("of the Eve of Preparation, of Passion"))
