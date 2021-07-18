import json
import os

import numpy as np
import pandas as pd
import pyrebase
from dotenv import load_dotenv, find_dotenv


class FirebaseDBManager:

    def __init__(self, firebase_config_path=None):
        if firebase_config_path is None:
            load_dotenv(find_dotenv())
            firebase_config_path = os.environ['FIREBASE_CONFIG_PATH']
        with open(firebase_config_path) as JSON:
            firebase_config = json.loads(JSON.read())
        self._db = pyrebase.initialize_app(firebase_config).database()

    def set_main_api_connection(self, public_url, secret_key):
        self._db.child('main_api_connection').child('public_url').set(public_url)
        self._db.child('main_api_connection').child('secret_key').set(secret_key)

    def get_main_api_connection(self):
        public_url = self._db.child('main_api_connection').child('public_url').get().val()
        secret_key = self._db.child('main_api_connection').child('secret_key').get().val()
        return public_url, secret_key

    def push_faq(self, faq: dict):
        assert 'question' in faq and isinstance(faq['question'], str)
        assert 'answer' in faq and isinstance(faq['answer'], str)
        self._db.child('FAQ').push(faq)

    def clear_all_faqs(self):
        self._db.child('FAQ').remove()

    def get_all_faqs(self) -> pd.DataFrame:
        records = list(self._db.child('FAQ').get().val().items())
        return pd.DataFrame([r[1] for r in records])

    def get_faq_by_question(self, question: str) -> dict:
        record = self._db.child('FAQ').order_by_child('question').equal_to(question).get().val()
        record = list(dict(record).items())
        if not record:
            return dict()
        return record[0][1]

    def push_feedback(self, user_feedback: dict) -> pd.DataFrame:
        assert 'user_question' in user_feedback and isinstance(user_feedback['user_question'], str)
        assert 'related' in user_feedback and isinstance(user_feedback['related'], bool)
        assert 'nearest_faq' in user_feedback
        assert 'question' in user_feedback['nearest_faq']
        assert isinstance(user_feedback['nearest_faq']['question'], str)
        assert isinstance(user_feedback['nearest_faq']['answer'], str)
        assert isinstance(user_feedback['nearest_faq']['score'], (np.float32, float))
        self._db.child('Feedback').push(user_feedback)

    def get_all_feedback(self):
        records = list(self._db.child('Feedback').get().val().items())
        return pd.DataFrame([r[1] for r in records])


if __name__ == "__main__":
    feedback = {
        "user_question": "Can children get vaccinated",
        "nearest_faq": {
            "question": "Is it safe for children to take vaccine?",
            "answer": "Hell no, shut up!",
            "score": 0.98999
        },
        "related": True
    }
    firebase_db = FirebaseDBManager()
    firebase_db.push_feedback(feedback)
