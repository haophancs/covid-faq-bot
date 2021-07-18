# COVID FAQ Bot
Source code of Telegram Bot of a Question-Answering system about COVID-19 content

**Author:**
- Hao Phu Phan (Фан Фу Хао, БПМИ208)
- ffan@edu.hse.ru  
- Higher School of Economics, Moscow

## Overview
### System structure
Including 4 main modules:
- *Firebase realtime database:* for storing FAQ data and user behaviors
- *Question-Answering:* returning the most related FAQ by the given input question
- *Main API:* communicating with database and running Question-Answering, powered by Flask
- *Telegram Bot*: used as a front-end module to get and answer user's question, as well as collecting their behaviors (so far only feedbacks collected)
### QA solution
By the given question from user, using Semantic Textual Similarity module to get the nearest FAQ. And then show the answer of that FAQ
#### Semantic Textual Similarity
- Powered by transformers architecture (library from HuggingFace)
- Selected pretrained model is ```covid-twitter-bert``` from Digital Epidemiology Lab EPFL, a transformer-based model, pretrained on a large corpus of Twitter messages on the topic of COVID-19
- Transformers model embedding user's question text to vector
- Using _cosine similarity_ to calculating the similarity score between the vector of each FAQ in database and the vector of user's question and return the FAQ having highest score


## Getting started (Linux)
### Install
```
conda create --name covid-faq-bot python=3.7
conda activate covid-faq-bot
python init.py --device-type cpu # (gpu, cpu)
```

### Necessary keys
- ```config/api.json```: ```secret_key``` - self-defined key
- ```config/telebot.json```: ```bot_name```, ```api_key``` - Telegram bot's name and key (Make sure to ```/setinline``` the bot)
- ```config/firebase.json```: **(anyway, it's no problem to use my already available config)** On firebase console, create a project, then create a web app and see "Firebase realtime database" in detailed configurations 

Optional
- ```config/api.json```: ```ngrok_auth_token``` - Ngrok account's auth token when ```run_with_ngrok``` is ```true```

### Run
Crawl FAQs from WHO official page (by ```beautifulsoup4```) and push them to Firebase database
```shell
python update_faqs.py
```
Start main API
```shell
python src/api.py
```
Open new session in terminal and start Telegram bot (Make sure project's virtual environment activated)
```shell
python src/telebot.py
```

## Detailed Configurations 
### Main API 
Located at ```config/api.json```
- ```secret_key```: self-defined key, SHA256-generated code recommended
- ```run_with_ngork```: use or not use Ngrok to make tunnel to the localhost (```true``` or ```false```)  
- ```ngrok_auth_token```: Ngrok account's auth token, provided if ```run_with_ngrok``` is ```true```
- ```port```: port of localhost, 5001 by default
- ```similarity_threshold```: threshold of STS score to determine if the nearest FAQ is related to the given input question or not
### Semantic Textual Similarity
Located at ```config/sts/config.json```
- ```device```: device used by model (```cpu```, ```cuda:0```,...)
- ```selected_pretrained_model```: ```covid-twitter-bert``` by default
- ```embedding_type```: ```text-vector1d``` by default, embedding text to vector
- ```embedding_type```: type of vector normalization, ```l2``` by default
### Firebase realtime database
Located at ```config/firebase.json```. 
- Paste the web app's config from Firebase project's console here
- Add one more argument: ```database_url```: URL of created Firebase realtime database of the project on Firebase console
### Telegram bot
Located at ```config/firebase.json```.
- Bot created by Telegram BotFather, **inline queries enabled**
- ```bot_name```: Bot's name
- ```api_key```: Bot's API key
- ```start_text```: text to show for command ```/start``` 
- ```help_text```: text to show for command ```/help```
- ```max_other_faqs```: number of other related FAQs beside the nearest FAQ, for recommendation purpose
- ```main_api_connection```: specification to make connection with main API. There are 3 options:
    - ```default```: get URL and secret key to connect main API from main API config in the same project root folder
    - ```check_db```: get URL and secret to connect main API from Firebase database. In fact, when main API is ready, it will update the these values on Firebase database
    - Manual specification: ```{"url": <main_api_url>, "secret_key": <main_api_secret_key>}```

## Other notes
### Quick deployment using Google Colab
For utilizing GPU

**Main API**
- Create notebook with GPU on Google Colab, clone and install the project
- Set ```run_with_ngrok``` as ```true``` in main API config
- Set ```ngrok_auth_token``` as Ngrok account's auth token in main API config
- Run main API: ```python src/api.py```

**Telegram bot**  
- Clone, install the project on another machine (or even another Colab notebook)
- Set ```main_api_connection``` as ```check_db``` in Telegram bot config
- Run Telegram Bot: ```python src/telebot.py```

### Demo
Video: https://drive.google.com/file/d/1OmRSOOqYhKKa5BwSfNaXR-DhweRRMzhB/view?usp=sharing