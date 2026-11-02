# Summarises page, word and language-block counts by newspaper so the supplied tracker can be checked against the archive description.
import pandas as pd

t = pd.read_csv("data/LanguageSplitTracker70-94.csv")
t["source"] = t["Edition"].str.extract(r"^([A-Za-z]+)_")
g = t.groupby("source").agg(pages=("Edition", "count"), total_words=("Tot_Wordcount", "sum"),
                            xh_words=("Xho_Wordcount", "sum"), en_words=("Eng_Wordcount", "sum"),
                            blocks=("Tot_Blocks", "sum"), xh_blocks=("Xh_Blocks", "sum"))
g["unlabelled_words"] = g["total_words"] - g["xh_words"] - g["en_words"]
g["xh_word_share_%"] = (100 * g["xh_words"] / (g["xh_words"] + g["en_words"])).round(1)
g["xh_block_share_%"] = (100 * g["xh_blocks"] / g["blocks"]).round(1)
print(g.T)
