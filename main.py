import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


PATH_TO_ANNOTATION = "c:\\Users\\Acer\\Documents\\py_lab_2\\annotations_1.csv"

def create_dataframe(annotation_path: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=["Тип рецензии", "Текст рецензии", "Количество слов"])
    with open(annotation_path, mode="r", encoding="utf-8") as ann:
        for line in ann.readlines():
            line = line.split(",")
            with open(line[0], mode="r", encoding="utf-8") as file:
                film_name = file.readline().rstrip()
                text = "\n".join(file.readlines())
                row = pd.Series({'Тип рецензии': line[2].rstrip(),'Текст рецензии': text,'Количество слов': len(text.split(" "))}, name = film_name)
                df_new_row = pd.DataFrame([row], columns=df.columns)
                df = pd.concat([df, df_new_row])
    df.dropna()
    return df


def get_static_info(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe()


def sort_dataframe_by_word_count(df: pd.DataFrame, count: int) -> pd.DataFrame:
    return df[df["Количество слов"] <= count]


def sort_dataframe_by_mark(df: pd.DataFrame, mark: str) -> pd.DataFrame:
    return df[df["Тип рецензии"] == mark]


def stats_for_marks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop('Текст рецензии', axis=1)
    df = df.groupby('Тип рецензии')

    df_max = df.max().values.tolist()
    df_min = df.min().values.tolist()
    df_mean = df.mean().values.tolist()

    df_max = sum(df_max, [])
    df_min = sum(df_min, [])
    df_mean = sum(df_mean, [])

    result = pd.DataFrame({'Тип рецензии': ['bad', 'good']})
    result["max"] = df_max
    result["min"] = df_min
    result["mean"] = df_mean
    
    return result


def get_hist(df: pd.DataFrame, mark: str) -> pd.DataFrame:
    stemmer = SnowballStemmer("russian")
    res = {}
    for text in df[df["Тип рецензии"] == mark]['Текст рецензии']:
        tokens = word_tokenize(text)
        lemmatized_words = [stemmer.stem(word) for word in tokens]
        list_to_dict(lemmatized_words, res)
    return pd.DataFrame(res, index=[0])


def list_to_dict(a: list, b: dict) -> dict:
    for i in a:
        if i in b.keys():
            b[i] +=1
        else:
            b[i] =1
    return b



if __name__ == "__main__":
    
    df = create_dataframe(PATH_TO_ANNOTATION)
    print(get_hist(df, "good"))