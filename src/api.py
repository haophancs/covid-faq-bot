import json
import os

from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, jsonify
from pyngrok import ngrok

from data.firebase import FirebaseDBManager

app = Flask(__name__)
load_dotenv(find_dotenv())
with open(os.environ['API_CONFIG_PATH']) as JSON:
    app_config = json.loads(JSON.read())
    app.secret_key = app_config['secret_key']

firebase_db = FirebaseDBManager()
sts = None
faq_data = None


@app.route('/')
def home():
    return jsonify({"message": "Hello World! I am COVID FAQ bot"}), 200


@app.route('/all-faq', methods=['GET'])
def all_faq():
    data = request.args.to_dict()
    assert 'secret_key' in data
    if data['secret_key'] != app.secret_key:
        return jsonify({"message": "Wrong secret key"}), 401
    return jsonify(faq_data.to_dict(orient='records'))


@app.route('/faq', methods=['GET'])
def get_faq():
    data = request.args.to_dict()
    assert 'secret_key' in data
    if data['secret_key'] != app.secret_key:
        return jsonify({"message": "Wrong secret key"}), 401
    assert 'question' in data
    faq = firebase_db.get_faq_by_question(data['question'])
    if faq:
        return jsonify(faq)
    return jsonify({"message": "FAQ not found"}), 404


@app.route('/update-faq-set', methods=['POST'])
def update_faq_set():
    data = request.get_json(force=True)
    assert 'secret_key' in data
    if data['secret_key'] != app.secret_key:
        return jsonify({"message": "Wrong secret key"}), 401
    global faq_data
    faq_data = firebase_db.get_all_faqs()
    return jsonify({"message": "FAQ set updated"})


@app.route('/nearest-faq', methods=['GET'])
def nearest_faq():
    data = request.args.to_dict()
    assert 'secret_key' in data
    if data['secret_key'] != app.secret_key:
        return jsonify({"message": "Wrong secret key"}), 401
    assert 'question' in data

    result = firebase_db.get_faq_by_question(data['question'].strip())
    if not result:
        index, score = sts.get_stored_best_match(data['question'], return_indices=True)
        if score < app_config['similarity_threshold']:
            return jsonify({"message": "Related FAQ not found"}), 404
        result = faq_data.iloc[index]
        result['score'] = score
        result = result.to_dict()
    else:
        result['score'] = 1
    return jsonify({"nearest-faq": result})


@app.route('/n-nearest-faqs', methods=['GET'])
def n_nearest_faq():
    data = request.args.to_dict()
    assert 'secret_key' in data
    if data['secret_key'] != app.secret_key:
        return jsonify({"message": "Wrong secret key"}), 401
    assert 'question' in data and 'n-returns' in data
    indices, scores = sts.get_stored_best_matches(data['question'],
                                                  nbest=int(data['n-returns']),
                                                  return_indices=True)
    if firebase_db.get_faq_by_question(data['question'].strip()):
        scores[0] = 1
    if scores[0] < app_config['similarity_threshold']:
        return jsonify({"message": "Related FAQs not found"}), 404
    result = faq_data.iloc[indices]
    result.loc[:, 'score'] = scores
    result = result.to_dict(orient='records')
    return jsonify({"n-nearest-faqs": result})


@app.route('/send-feedback', methods=['POST'])
def send_feedback():
    data = request.get_json(force=True)
    assert 'secret_key' in data
    if data['secret_key'] != app.secret_key:
        return jsonify({"message": "Wrong secret key"}), 401
    assert 'feedback' in data
    firebase_db.push_feedback(data['feedback'])
    return jsonify({"message": "Feedback sent"})


if __name__ == '__main__':
    faq_data = firebase_db.get_all_faqs()
    if app_config['run_with_ngrok']:
        ngrok.set_auth_token(app_config['ngrok_auth_token'])
        http_tunnel = ngrok.connect(app_config['port'])
        print(http_tunnel)
        firebase_db.set_main_api_connection(http_tunnel.public_url, app_config['secret_key'])
    from sts import SemanticTextualSimilarityPipeline

    sts = SemanticTextualSimilarityPipeline(stored_texts=faq_data.question)
    app.run(host='localhost', port=app_config['port'])
