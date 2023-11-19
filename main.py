import os


import numpy as np
import pandas as pd
import matplotlib


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
    return df.loc[mark]


if __name__ == "__main__":
    df = create_dataframe(PATH_TO_ANNOTATION)
    df = sort_dataframe_by_mark(df, "1+1")
    print(df)