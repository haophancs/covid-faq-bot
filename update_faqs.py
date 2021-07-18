from src.data.firebase import FirebaseDBManager
from src.data.who import WHOFAQCrawler

if __name__ == "__main__":
    firebase_db = FirebaseDBManager()
    firebase_db.clear_all_faqs()
    qa_df = WHOFAQCrawler.faq_dataset()
    for faq in qa_df.to_dict(orient='records'):
        firebase_db.push_faq(faq)
