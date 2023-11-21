import string


import spacy
import pandas as pd
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from progress.bar import IncrementalBar
import multiprocessing




TRASH_LIST = [x for x in string.punctuation + string.digits]
TRASH_LIST += ["\n", "«", "»", '“', '”', '—', '-', '–']


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


def get_hist(df: pd.DataFrame, mark: str, n: int) -> pd.DataFrame:
    stemmer = SnowballStemmer("russian")
    list = []
    
    nlp = spacy.load("ru_core_news_md")
    stopwords_ru = stopwords.words("russian")
    stopwords_ru += df.index.to_list()
    stopwords_ru += ['серия', 'фильм', 'сезон', 'сериал', 'который']
    stopwords_ru += list(nlp.Defaults.stop_words)
    stopwords_ru = list(set(stopwords_ru))

    bar = IncrementalBar('Progress', max = round(len(df.index)/2))

    for text in sort_dataframe_by_mark(df, mark)['Текст рецензии']:
        bar.next()
        tokens = word_tokenize(text)
        lemmatized_words = [stemmer.stem(word) for word in tokens if word not in stopwords_ru and word not in string.punctuation + "«»–..."]
        list += lemmatized_words
    list = nltk.Text(list)
    bar.finish()
    return pd.Series(dict(FreqDist(list).most_common(n)))



def get_hist2(df: pd.DataFrame, mark: str, n: int) -> pd.Series:
    m = Mystem()
    

    nlp = spacy.load("ru_core_news_md")
    stopwords_ru = stopwords.words("russian")
    stopwords_ru += ['серия', 'фильм', 'сезон', 'сериал', 'который', 'первый', "второй", "персонаж", " ", '  ']
    stopwords_ru += list(nlp.Defaults.stop_words)
    stopwords_ru += [word.lower() for word in df.index]
    stopwords_ru = list(set(stopwords_ru))
    

    data = sort_dataframe_by_mark(df, mark)['Текст рецензии']
    with multiprocessing.Manager() as manager:
        d = manager.dict()
        p = multiprocessing.Pool(8)
        p.starmap_async(process, [(text, m, stopwords_ru, d, n, nlp) for text in data])
        p.close()
        p.join()
        return pd.Series(dict(dict_to_FreqDist(d).most_common(n)))

def process(text: str, m: Mystem, stopwords_ru: list, d: dict, n: int, nlp) -> None:
    text = del_trash(text)
    l = [word for word in m.lemmatize(text) if (word not in stopwords_ru) and nlp(word)[0].pos_ not in ["VERB", "NOUN"]]
    processed = FreqDist(nltk.Text(l)).most_common(n)
    merge(d, dict(processed))


def show_barh(df: pd.Series) -> None:
    plt.barh(df.index, df.values)
    plt.xlabel("Количество встреченных слов")
    plt.ylabel("Самые часто используемые слова")
    plt.title("Частотный анализ слов из рецензий")
    plt.show()


def list_to_dict(a: list, b: dict) -> dict:
    for i in a:
        if i in b.keys():
            b[i] +=1
        else:
            b[i] =1
    return b


def del_trash(text: str) -> str:
    res = ''
    for i in text:
        if i not in TRASH_LIST:
            res += i
    return res


def merge(a: dict, b: FreqDist) -> None:
    for key, value in b.items():
        if key in a.keys():
            a[key] += value
        else:
            a[key] = value


def dict_to_FreqDist(a: dict) -> FreqDist:
    b = FreqDist()
    for key, value in a.items():
        b[key] = value
    return b


if __name__ == "__main__":
    df = create_dataframe(PATH_TO_ANNOTATION)
    df = get_hist2(df, "bad", 65) 
    show_barh(df)
