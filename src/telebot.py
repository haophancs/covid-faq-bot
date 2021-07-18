import json
import logging
import os

import requests
from dotenv import find_dotenv, load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import *

from data.firebase import FirebaseDBManager

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

load_dotenv(find_dotenv())
with open(os.environ['TELEBOT_CONFIG_PATH']) as JSON:
    telebot_config = json.loads(JSON.read())

logging.info(f'Starting Bot..., name = {telebot_config["bot_name"]}')
if telebot_config['main_api_connection'] == 'default':
    with open(os.environ['API_CONFIG_PATH']) as JSON:
        main_api_config = json.loads(JSON.read())
        main_api_key = main_api_config['secret_key']
        main_api_url = 'http://localhost' + ":" + str(main_api_config['port'])
elif telebot_config['main_api_connection'] == 'check_db':
    firebase_db = FirebaseDBManager()
    main_api_url, main_api_key = firebase_db.get_main_api_connection()
else:
    main_api_key = telebot_config['main_api_connection']['secret_key']
    main_api_url = telebot_config['main_api_connection']['url']

cached_data_form = {
    "fdbk": {},
    "ofaqs": {},
    "pre_question": {}
}
cached_data = dict()

r = requests.get(
    main_api_url + '/',
    params={
        'secret_key': main_api_key
    }
)
if r.status_code == 200:
    logging.info(f'Connected to main api: url = {main_api_url}')


def start_command(update: Update, context: CallbackContext):
    update.message.reply_text(
        telebot_config['start_text'],
        parse_mode=telebot_config['text_parsing_mode']
    )


def help_command(update: Update, context: CallbackContext):
    update.message.reply_text(
        telebot_config['help_text'],
        parse_mode=telebot_config['text_parsing_mode']
    )


def covid_info_command(update: Update, context: CallbackContext):
    update.message.reply_text('This is a custom command, you can add whatever text you want here.')


def prepare_msg_main_keyboard(update: Update, need_feedback=True):
    keyboard = []
    if need_feedback:
        keyboard.append([
            InlineKeyboardButton(
                "Related \U0001f44d",
                callback_data=f"fdbk_pos_{update.message.message_id}"
            ),
            InlineKeyboardButton(
                "Not related \U0001f44e",
                callback_data=f"fdbk_neg_{update.message.message_id}"
            ),
        ])

    keyboard.append([
        InlineKeyboardButton(
            "Show other related FAQs",
            callback_data=f"ofaqs_{update.message.message_id}"
        )
    ])
    return keyboard


def cache_question_data(update: Update, nearest_faq=None, other_faq_questions=None):
    user_question = update.message.text
    if user_question.startswith(f"@{telebot_config['bot_name']}"):
        user_question = user_question[len(f"@{telebot_config['bot_name']}"):]
        user_question = user_question.lstrip(' ').rstrip('\n').rstrip(' ')
    assert (nearest_faq is None and other_faq_questions is None) \
           or (nearest_faq is not None and other_faq_questions is not None)
    if nearest_faq is None:
        cached_data[update.message.from_user.id]['pre_question'].update({
            user_question.lower(): "not supported"
        })
        return
    if nearest_faq['score'] < 1:
        cached_data[update.message.from_user.id]['fdbk'].update({
            update.message.message_id: {
                "user_question": user_question,
                "nearest_faq": nearest_faq,
            }
        })
    cached_data[update.message.from_user.id]['ofaqs'].update({update.message.message_id: other_faq_questions})
    cached_data[update.message.from_user.id]['pre_question'].update({
        user_question.lower(): {
            "nearest_faq": nearest_faq,
            "other_faq_questions": other_faq_questions
        }
    })


def load_cached_question_data(user_question: str, user_id):
    user_question = user_question.lower()
    nearest_faq = cached_data[user_id]['pre_question'][user_question]['nearest_faq']
    other_faq_questions = cached_data[user_id]['pre_question'][user_question]['other_faq_questions']
    return nearest_faq, other_faq_questions


def handle_message(update: Update, context: CallbackContext):
    if update.message.from_user.id not in cached_data:
        cached_data[update.message.from_user.id] = cached_data_form
    user_question = update.message.text
    if user_question.startswith(f"@{telebot_config['bot_name']}"):
        user_question = user_question[len(f"@{telebot_config['bot_name']}"):]
        user_question = user_question.lstrip(' ').rstrip('\n').rstrip(' ')
    logging.info(f'user ({update.message.chat.id}) asked: {user_question}')
    if user_question.lower() not in cached_data[update.message.from_user.id]['pre_question']:
        endpoint = "/n-nearest-faqs"
        response = requests.get(
            main_api_url + endpoint,
            params={
                'question': user_question,
                'n-returns': 1 + telebot_config['max_other_faqs'],
                'secret_key': main_api_key
            }
        )
        if response.status_code == 404:
            cache_question_data(update, None, None)
            update.message.reply_text("Sorry! I can't find any FAQ related to your question")
            return
        response = response.json()
        nearest_faq = response['n-nearest-faqs'][0]
        other_faq_questions = [item['question'] for item in response['n-nearest-faqs'][1:]]
        need_feedback = nearest_faq['score'] < 1
    elif cached_data[update.message.from_user.id]['pre_question'][user_question.lower()] == "not supported":
        update.message.reply_text("Sorry! I can't find any FAQ related to your question")
        return
    else:
        nearest_faq, other_faq_questions = load_cached_question_data(user_question, update.message.from_user.id)
        need_feedback = False
    cache_question_data(update, nearest_faq, other_faq_questions)
    if nearest_faq['score'] < 1:
        final_text = f'*Related FAQ: {nearest_faq["question"]}*\n\n{nearest_faq["answer"]}'
    else:
        final_text = f'*{nearest_faq["question"]}*\n\n{nearest_faq["answer"]}'
    update.message.reply_text(
        final_text,
        parse_mode=telebot_config['text_parsing_mode'],
        reply_markup=InlineKeyboardMarkup(prepare_msg_main_keyboard(update, need_feedback))
    )


def handle_button(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    if query.data.startswith("fdbk"):
        query.answer(text=f"Thanks for your feedback!")
        _, fdbk_type, msg_id = query.data.split('_')
        msg_id = int(msg_id)
        keyboard = [[InlineKeyboardButton("Show other similar FAQs", callback_data=f"ofaqs_{msg_id}")]]
        query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(keyboard))
        fdbk_data = cached_data[query.from_user.id]['fdbk'][msg_id]
        cached_data[query.from_user.id]['fdbk'].pop(msg_id)
        if fdbk_type == 'pos':
            fdbk_data['related'] = True
        else:
            fdbk_data['related'] = False
        r = requests.post(
            main_api_url + "/send-feedback",
            json={
                'secret_key': main_api_key,
                'feedback': fdbk_data
            }
        )
        if r.status_code != 200:
            logging.error(f'Request to sent feedback failed, status code = {r.status_code}')

    elif query.data.startswith('ofaqs'):
        _, msg_id = query.data.split('_')
        msg_id = int(msg_id)
        keyboard = []
        for q_idx, question in enumerate(cached_data[query.from_user.id]['ofaqs'][msg_id]):
            keyboard.append([
                InlineKeyboardButton(
                    question,
                    switch_inline_query_current_chat=question
                )
            ])
        query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(keyboard))


def error(update: Update, context: CallbackContext):
    logging.error(f'Update {update} caused error {context.error}')


if __name__ == '__main__':
    updater = Updater(telebot_config['api_key'], use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('start', start_command))
    dp.add_handler(CommandHandler('help', help_command))
    dp.add_handler(CommandHandler('covid_info', covid_info_command))
    dp.add_handler(MessageHandler(Filters.text, handle_message))
    dp.add_handler(CallbackQueryHandler(handle_button))
    dp.add_error_handler(error)
    updater.start_polling()
    updater.idle()
