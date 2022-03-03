# English to Assyrian/Eastern Syriac Model

This is an English to Assyrian/Eastern Syriac machine translation model, it uses [English to Arabic](https://huggingface.co/Helsinki-NLP/opus-mt-en-ar) model as the base model.
The source code is well documented, and was made such that it can be read by inexperienced developers.

Although the project aim is to Build a English to Assyrian - the ones that fall under [Northeastern Neo-Aramaic](https://en.wikipedia.org/wiki/Northeastern_Neo-Aramaic) -  the current model mostly provides translation for Classical Syriac. This model is a good initial step, but I hope future work will make it more inline with Assyrian dialects.

*Please note that Assyrian and Easter Syriac are used interchangeably.*

## To Use

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("mt-empty/english-assyrian")

model = AutoModelForSeq2SeqLM.from_pretrained("mt-empty/english-assyrian")

translator = pipeline("translation", model=model, tokenizer=tokenizer)

print("tomorrow morning", translator("tomorrow morning"))

```

[test.py](./test.py)/[test.ipynb](./test.ipynb) contains examples on how to use the translation pipeline.

## Dataset

The dataset are sourced from:

* [Classical Syriac bible](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/en-syr)
* [pericopes](https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:114404/tab/2)

## Tokenizer

[SentencePiece](https://github.com/google/sentencepiece/) was used for tokenization.

## Evaluation

[SacreBlue](https://github.com/mjpost/sacrebleu) was used for evaluation. Running it for 50 epochs produced a score of **33**.

## To Train

Please make sure you have installed all the required dependencies,
Run `python model.py`, it will train for `50` epochs, this can be changed in the code.

## Before Contributing

This project utilizes pre-commit hooks, so please run the following before submitting a pull request:

1. Install requirements, `pip install -r requirements.txt`
2. Configure pre-commit hooks, `pre-commit install`
3. (Optional) Run hooks manually, `pre-commit run --all-files`
4. Submit a pull request

### Main Dependencies

```
datasets
transformers
sentencepiece
pandas
pytorch
sacrebleu
```

Please install the appropriate version of [pytorch](https://pytorch.org/) for your machine, `cuda` is needed if you want to train on GPU.

## Interesting Research Projects

* [North Eastern Neo-Aramaic Database](https://github.com/CambridgeSemiticsLab)
* [Neo-Aramaic Web Corpora: Christian Urmi and Turoyo](http://neo-aramaic.web-corpora.net/index_en.html)
