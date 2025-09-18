import os 
import pandas as pd

def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "../" "data")

    english_file = os.path.join(data_dir, "english.txt")
    urdu_file = os.path.join(data_dir, "urdu.txt")

    with open(english_file, "r", encoding="utf-8") as f:
        english_corpus = [line.strip() for line in f]
        english_corpus = list(english_corpus)

    with open(urdu_file, "r", encoding="utf-8") as f:
        urdu_corpus = [line.strip() for line in f]
        urdu_corpus = list(urdu_corpus)

    data = pd.DataFrame({'english': english_corpus, 'urdu': urdu_corpus})
    return data
