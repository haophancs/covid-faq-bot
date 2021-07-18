import json
import os
import re
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv


class WHOFAQCrawler:
    @staticmethod
    def default_source_urls():
        load_dotenv(find_dotenv())
        with open(os.environ['WHO_FAQ_URLS_PATH']) as JSON:
            return json.loads(JSON.read())['source_urls']

    @staticmethod
    def preprocess(text: str) -> str:
        punctuations = ['.', ',', '!', '?', ';', ':', '\"', '\'', ')', '(']
        for p in punctuations:
            text = text.replace(f'\n{p}', p)
            text = re.sub(r'\t', ' ', text)
            text = re.sub(r' {2}', ' ', text)
            text = re.sub(r'\n\n', '\n', text)
            text = re.sub(r'\n ', '\n', text)
            text = re.sub(r' \n', '\n', text)
        return text

    @staticmethod
    def crawl_qas(url: str) -> (List, List):
        response = requests.get(url)
        txt = response.text
        soup = BeautifulSoup(txt, 'html.parser')
        questions = soup.find_all("a", class_="sf-accordion__link")
        answers = soup.find_all("p", class_="sf-accordion__summary")
        return questions, answers

    @staticmethod
    def faq_dataset(source_urls: List = None) -> pd.DataFrame:
        if not source_urls:
            source_urls = WHOFAQCrawler.default_source_urls()
        qa_data = {
            "question": [],
            "answer": []
        }
        for url in source_urls:
            questions, answers = WHOFAQCrawler.crawl_qas(url)
            assert len(questions) == len(answers)
            qa_data['question'] += questions
            qa_data['answer'] += answers
        qa_df = pd.DataFrame(qa_data)
        qa_df.question = qa_df.question.apply(
            lambda q: WHOFAQCrawler.preprocess(q.get_text(' ', strip=True))
        )
        qa_df.answer = qa_df.answer.apply(
            lambda a: WHOFAQCrawler.preprocess(a.get_text('\n', strip=True))
        )
        return qa_df.drop_duplicates(subset=['question'])
