import os

import pandas as pd
from nltk.corpus import stopwords

# nltk.download('stopwords')

"""
Generates all non stop english vocabulary, then outputs the english vocabulary size
This size is used to estimate the English+Assyrian vocabulary size used for the tokenizer
Rough calculation, English+Assyrian: English size * 1.8 (because Assyrian tend to be smaller)
"""

stop_words = set(stopwords.words("english"))

src_file = "dataset/pericopes_data.csv"
vocab_file = "dataset/vocab.txt"
temp_file = "temp.csv"
df1 = pd.read_csv(src_file, usecols=[0], index_col=False)

df1 = df1["of the Unleavened Bread"].apply(
    lambda x: " ".join([word for word in x.split() if word not in (stop_words)])
)
print(src_file)
print(df1)

# df1.to_csv(temp_file, index=False)


src_file = "dataset/bible_data.csv"
df2 = pd.read_csv(src_file, usecols=[0], index_col=False)
df2 = df2[
    "The book of the genealogy of Jesus Christ , the son of David, the son of Abraham."
].apply(lambda x: " ".join([word for word in x.split() if word not in (stop_words)]))
print(src_file)
print(df2)

pd.concat([df1, df2]).to_csv(temp_file, index=False)

# only works with linux systems
os.system(
    f'tr -sc "A-Za-z" "\n" < {temp_file} | tr "A-Z" "a-z" | sort | uniq -c | sort -n -r > {vocab_file}'
)
os.system(f"rm {temp_file}")
os.system(f"wc -l {vocab_file}")
