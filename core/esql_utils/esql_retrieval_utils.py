import os
import json
import string
import logging
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from typing import List


def nltk_downloads():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


def save_dataframe_to_csv(df: pd.DataFrame, path: str):
    try:
        df.to_csv(path, index=False)
        print(f"DataFrame saved successfully to {path}")
    except Exception as e:
        logging.error(f"An error occurred while saving the DataFrame: {e}")


def clean_text(textData: str) -> str:
    if isinstance(textData, str):
        textData = textData.lower()
        textData = textData.replace("       ", '')
        textData = textData.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        stopWordsSet = set(stopwords.words('english'))
        tokens = word_tokenize(textData)
        tokens = [token for token in tokens if not token.lower() in stopWordsSet]
        processedTextData = ' '.join(tokens)
        return processedTextData
    else:
        return ''


def construct_column_information(table_desc_df: pd.DataFrame, table_name: str) -> pd.Series:
    def build_column_info(row):
        column_info = f"The information about the {row['original_column_name']} column of the {table_name} table [{table_name}.{row['original_column_name']}] is as following."
        if pd.notna(row['column_description']):
            column_info += f" The {row['original_column_name']} column can be described as {row['column_description']}."
        if pd.notna(row['value_description']):
            column_info += f" The value description for the {row['original_column_name']} is {row['value_description']}"
        column_info = column_info.replace("       ", ' ')
        column_info = column_info.replace("       ", ' ')
        return column_info
    column_info_series = table_desc_df.apply(build_column_info, axis=1)
    return column_info_series


def process_database_descriptions(database_description_path: str):
    all_column_infos = []
    for filename in os.listdir(database_description_path):
        if filename.endswith(".csv") and filename != "db_description.csv":
            print(f"------> {filename} table start to be processed.")
            file_path = os.path.join(database_description_path, filename)
            try:
                df = pd.read_csv(file_path)
            except:
                df = pd.read_csv(file_path, encoding='ISO-8859-1')
            table_name = filename.replace('.csv', '')
            column_info_series = construct_column_information(df, table_name)
            column_info_df = column_info_series.to_frame(name='column_info')
            all_column_infos.append(column_info_df)
    all_info_df = pd.concat(all_column_infos, ignore_index=True)
    output_path = os.path.join(database_description_path, 'db_description.csv')
    all_info_df.to_csv(output_path, index=False)
    print(f"---> Database information saved successfully to {output_path}")


def process_all_dbs(dataset_path: str, mode: str):
    nltk_downloads()
    databases_path = dataset_path + f"/{mode}/{mode}_databases"
    for db_directory in os.listdir(databases_path):
        if db_directory == ".DS_Store":
            continue
        print(f"----------> Start to process {db_directory} database.")
        db_description_path = databases_path + "/" + db_directory + "/database_description"
        process_database_descriptions(database_description_path=db_description_path)
    print("\n\n All databases processed and db_description.csv files are created for all.\n\n")


def get_relevant_db_descriptions(database_description_path: str, question: str, relevant_description_number: int = 6) -> List[str]:
    db_description_csv_path = database_description_path + "/db_description.csv"
    if not os.path.exists(db_description_csv_path):
        process_database_descriptions(database_description_path)
    db_desc_df = pd.read_csv(db_description_csv_path)
    db_description_corpus = db_desc_df['column_info'].tolist()
    db_description_corpus_cleaned = [clean_text(description) for description in db_description_corpus]
    tokenized_db_description_corpus_cleaned = [doc.split(" ") for doc in db_description_corpus_cleaned]
    bm25 = BM25Okapi(tokenized_db_description_corpus_cleaned)
    tokenized_question = question.split(" ")
    relevant_db_descriptions = bm25.get_top_n(tokenized_question, db_description_corpus, n=relevant_description_number)
    return relevant_db_descriptions


def get_db_column_meanings(database_column_meaning_path: str, db_id: str) -> List[str]:
    with open(database_column_meaning_path, 'r') as file:
        column_meanings = json.load(file)
    meanings = []
    for key, explanation in column_meanings.items():
        if key.startswith(db_id + "|"):
            _, table_name, column_name = key.split("|")
            meaning = f"# Meaning of {column_name} column of {table_name} table in database is that {explanation.strip('# ').strip()}"
            meanings.append(meaning)
    return meanings