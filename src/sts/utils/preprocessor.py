"""
Following source code of paperwork
"UIT-HSE at WNUT-2020 Task 2: Exploiting CT-BERT for
Identifying COVID-19 Information on the Twitter Social Network"
- Authors: Khiem Tran, Hao Phan (Student БПМИ208 HSE), Kiet Nguyen, Ngan Luu Thuy Nguyen
- Paper link: https://aclanthology.org/2020.wnut-1.53/
- Source link: https://github.com/haophancs/transformers-exptool
"""

from nltk.tokenize import TweetTokenizer
from emoji import demojize, emoji_count
import pandas as pd
import re
import html
import unicodedata
import unidecode
import preprocessor as p
from ekphrasis.classes.segmenter import Segmenter
import contractions


class TextPreprocessor:
    _seg_tw = Segmenter(corpus="twitter")
    _w_tokenizer = TweetTokenizer()
    _control_char_regex = re.compile(r'[\r\n\t]+')
    _transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–ー-", u"'''\"\"---")])

    @staticmethod
    def normalize_punctuation(norm_text):
        # handle punctuation
        norm_text = norm_text.translate(TextPreprocessor._transl_table)
        norm_text = norm_text.replace('…', '...')
        norm_text = ''.join([unidecode.unidecode(t) if unicodedata.category(t)[0] == 'P' else t for t in norm_text])
        if '...' not in norm_text:
            norm_text = norm_text.replace('..', ' ... ')
        return norm_text

    @staticmethod
    def normalize_special_characters(norm_text):
        norm_text = re.sub(r"\x89Û_", "", norm_text)
        norm_text = re.sub(r"\x89ÛÒ", "", norm_text)
        norm_text = re.sub(r"\x89ÛÓ", "", norm_text)
        norm_text = re.sub(r"\x89ÛÏWhen", "When", norm_text)
        norm_text = re.sub(r"\x89ÛÏ", "", norm_text)
        norm_text = re.sub(r"China\x89Ûªs", "China's", norm_text)
        norm_text = re.sub(r"let\x89Ûªs", "let's", norm_text)
        norm_text = re.sub(r"\x89Û÷", "", norm_text)
        norm_text = re.sub(r"\x89Ûª", "", norm_text)
        norm_text = re.sub(r"\x89Û\x9d", "", norm_text)
        norm_text = re.sub(r"å_", "", norm_text)
        norm_text = re.sub(r"\x89Û¢", "", norm_text)
        norm_text = re.sub(r"\x89Û¢åÊ", "", norm_text)
        norm_text = re.sub(r"fromåÊwounds", "from wounds", norm_text)
        norm_text = re.sub(r"åÊ", "", norm_text)
        norm_text = re.sub(r"åÈ", "", norm_text)
        norm_text = re.sub(r"JapÌ_n", "Japan", norm_text)
        norm_text = re.sub(r"Ì©", "e", norm_text)
        norm_text = re.sub(r"å¨", "", norm_text)
        norm_text = re.sub(r"SuruÌ¤", "Suruc", norm_text)
        norm_text = re.sub(r"åÇ", "", norm_text)
        norm_text = re.sub(r"å£3million", "3 million", norm_text)
        norm_text = re.sub(r"åÀ", "", norm_text)
        norm_text = html.unescape(norm_text)
        return norm_text

    @staticmethod
    def normalize_contractions(norm_text):
        # Contractions
        norm_text = re.sub(r"don\x89Ûªt", "do not", norm_text)
        norm_text = re.sub(r"I\x89Ûªm", "I am", norm_text)
        norm_text = re.sub(r"you\x89Ûªve", "you have", norm_text)
        norm_text = re.sub(r"it\x89Ûªs", "it is", norm_text)
        norm_text = re.sub(r"doesn\x89Ûªt", "does not", norm_text)
        norm_text = re.sub(r"It\x89Ûªs", "It is", norm_text)
        norm_text = re.sub(r"Here\x89Ûªs", "Here is", norm_text)
        norm_text = re.sub(r"I\x89Ûªve", "I have", norm_text)
        norm_text = re.sub(r"can\x89Ûªt", "cannot", norm_text)
        norm_text = re.sub(r"That\x89Ûªs", "That is", norm_text)
        norm_text = re.sub(r"that\x89Ûªs", "that is", norm_text)
        norm_text = re.sub(r"This\x89Ûªs", "This is", norm_text)
        norm_text = re.sub(r"this\x89Ûªs", "this is", norm_text)
        norm_text = re.sub(r"You\x89Ûªre", "You are", norm_text)
        norm_text = re.sub(r"Don\x89Ûªt", "Do not", norm_text)
        norm_text = re.sub(r"Can\x89Ûªt", "Cannot", norm_text)
        norm_text = re.sub(r"you\x89Ûªll", "you will", norm_text)
        norm_text = re.sub(r"I\x89Ûªd", "I would", norm_text)
        norm_text = re.sub(r"donå«t", "do not", norm_text)

        norm_text = re.sub(r"He's", "He is", norm_text)
        norm_text = re.sub(r"She's", "She is", norm_text)
        norm_text = re.sub(r"It's", "It is", norm_text)
        norm_text = re.sub(r"he's", "he is", norm_text)
        norm_text = re.sub(r"she's", "she is", norm_text)
        norm_text = re.sub(r"it's", "it is", norm_text)

        norm_text = re.sub(r"He ain't", "He is not", norm_text)
        norm_text = re.sub(r"She aint't", "She is not", norm_text)
        norm_text = re.sub(r"It aint't", "It is not", norm_text)
        norm_text = re.sub(r"he aint't", "he is not", norm_text)
        norm_text = re.sub(r"she aint't", "she is not", norm_text)
        norm_text = re.sub(r"it aint't", "it is not", norm_text)
        norm_text = contractions.fix(norm_text)
        return norm_text

    @staticmethod
    def normalize_abbreviations(norm_text):
        norm_text = re.sub(r'R\.I\.P', 'Rest In Peace', norm_text)
        norm_text = re.sub(r'R\.i\.p', 'Rest in peace', norm_text)
        norm_text = re.sub(r'r\.i\.p', 'rest in peace', norm_text)
        norm_text = re.sub(r"U\.S", "United States", norm_text)
        norm_text = re.sub(r"u\.s", "united states", norm_text)
        norm_text = re.sub(r"w/e", "whatever", norm_text)
        norm_text = re.sub(r"w/", "with", norm_text)
        norm_text = re.sub(r"USAgov", "USA government", norm_text)
        norm_text = re.sub(r"usagov", "usa government", norm_text)
        norm_text = re.sub(r"recentlu", "recently", norm_text)
        norm_text = re.sub(r"Ph0tos", "Photos", norm_text)
        norm_text = re.sub(r"ph0tos", "photos", norm_text)
        norm_text = re.sub(r"amirite", "am I right", norm_text)
        norm_text = re.sub(r"exp0sed", "exposed", norm_text)
        norm_text = re.sub(r"<3", "love", norm_text)
        norm_text = re.sub(r"amageddon", "armageddon", norm_text)
        norm_text = re.sub(r"Trfc", "Traffic", norm_text)
        norm_text = re.sub(r"trfc", "traffic", norm_text)
        norm_text = re.sub(r"([0-9]+)(yr)", r"\1 years", norm_text)
        norm_text = re.sub(r"lmao", "laughing my ass off", norm_text, flags=re.I)
        norm_text = re.sub(r"lol", "laughing out loud", norm_text, flags=re.I)
        norm_text = re.sub(r"TRAUMATISED", "traumatized", norm_text)
        norm_text = re.sub(r"traumatised", "traumatized", norm_text)
        norm_text = re.sub(r"ppl", "people", norm_text)
        norm_text = re.sub(r"Ppl", "People", norm_text)
        norm_text = re.sub(r"sh\*t", r"shit", norm_text)
        norm_text = norm_text.replace("cv19", "COVID 19")
        norm_text = norm_text.replace("cvid19", "COVID 19")
        return norm_text

    @staticmethod
    def normalize_hashtag(norm_text):
        for hashtag in re.findall(r"#(\w+)", norm_text):
            norm_text = norm_text.replace(f'#{hashtag}', '#' + TextPreprocessor._seg_tw.segment(hashtag))
        return norm_text

    @staticmethod
    def normalize_token(token, keep_emojis=True, username="@USER", httpurl="httpurl"):
        lowercase_token = token.lower()
        if token.startswith("@"):
            return username
        elif lowercase_token.startswith("http") or lowercase_token.startswith("www"):
            return httpurl
        elif len(token) == 1:
            if keep_emojis:
                demojized = demojize(token)
                if ":regional_indicator_symbol_letter_" in demojized:
                    return ""
                if ":globe" in demojized:
                    return ":globe:"
                return demojized
            elif emoji_count(token) > 0:
                return ""
        return token

    @staticmethod
    def replace_multi_occurrences(norm_text, filler):

        # only run if we have multiple occurrences of filler
        if norm_text.count(filler) <= 1:
            return norm_text
        # pad fillers with whitespace
        norm_text = norm_text.replace(f'{filler}', f' {filler} ')
        # remove introduced duplicate whitespaces
        norm_text = ' '.join(norm_text.split())
        # find indices of occurrences
        indices = []
        for m in re.finditer(r'{}'.format(filler), norm_text):
            index = m.start()
            indices.append(index)
        # collect merge list
        merge_list = []
        old_index = None
        for i, index in enumerate(indices):
            if i > 0 and index - old_index == len(filler) + 1:
                # found two consecutive fillers
                if len(merge_list) > 0 and merge_list[-1][1] == old_index:
                    # extend previous item
                    merge_list[-1][1] = index
                    merge_list[-1][2] += 1
                else:
                    # create new item
                    merge_list.append([old_index, index, 2])
            old_index = index
        # merge occurrences
        if len(merge_list) > 0:
            new_text = ''
            pos = 0
            for (start, end, count) in merge_list:
                new_text += norm_text[pos:start]
                new_text += f'{count} {filler}'
                pos = end + len(filler)
            new_text += norm_text[pos:]
            norm_text = new_text
        return norm_text

    @staticmethod
    def normalize_text(norm_text,
                       config=None,
                       to_ascii=True,
                       to_lower=False,
                       keep_emojis=True,
                       segment_hashtag=True,
                       username="@USER",
                       httpurl="httpurl") -> str:
        if config is not None:
            to_lower = config['to_lower']
            to_ascii = config['to_ascii']
            keep_emojis = config['keep_emojis']
            username = config['username']
            httpurl = config['httpurl']
            segment_hashtag = config['segment_hashtag']

        if to_lower:
            norm_text = norm_text.lower()

        norm_text = TextPreprocessor.normalize_special_characters(norm_text)
        norm_text = TextPreprocessor.normalize_punctuation(norm_text)
        norm_text = TextPreprocessor.normalize_contractions(norm_text)
        norm_text = TextPreprocessor.normalize_abbreviations(norm_text)

        tokens = TextPreprocessor._w_tokenizer.tokenize(norm_text)
        norm_text = " ".join([TextPreprocessor.normalize_token(token,
                                                               keep_emojis=keep_emojis,
                                                               username=username,
                                                               httpurl=httpurl) for token in tokens])
        norm_text = TextPreprocessor.replace_multi_occurrences(norm_text, username)
        norm_text = TextPreprocessor.replace_multi_occurrences(norm_text, httpurl)

        norm_text = re.sub(r"(covid.19)", "COVID 19 ", norm_text, flags=re.I)
        norm_text = re.sub(r"(covid...19)", "COVID 19 ", norm_text, flags=re.I)
        norm_text = re.sub(r"covid19", " COVID 19 ", norm_text, flags=re.I)
        norm_text = re.sub(r"# COVID19", "#COVID 19", norm_text, flags=re.I)
        norm_text = re.sub(r"# COVID19", "#COVID 19", norm_text, flags=re.I)
        norm_text = re.sub(r'\s+', ' ', norm_text).strip()

        if segment_hashtag:
            norm_text = TextPreprocessor.normalize_hashtag(norm_text)
        p.set_options(p.OPT.RESERVED, p.OPT.SMILEY)
        norm_text = p.clean(norm_text)

        # replace \t, \n and \r characters by a whitespace
        norm_text = re.sub(TextPreprocessor._control_char_regex, ' ', norm_text)
        # remove all remaining control characters
        norm_text = ''.join(ch for ch in norm_text if unicodedata.category(ch)[0] != 'C')

        norm_text = re.sub(r" p \. m \.", "  p.m.", norm_text, flags=re.I)
        norm_text = re.sub(r" p \. m ", " p.m ", norm_text, flags=re.I)
        norm_text = re.sub(r" a \. m \.", "  a.m.", norm_text, flags=re.I)
        norm_text = re.sub(r" a \. m ", " a.m ", norm_text, flags=re.I)
        norm_text = re.sub(r"'s", " 's ", norm_text)
        norm_text = re.sub(r"(covid.19)", "COVID19", norm_text, flags=re.I)

        norm_text = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", norm_text)
        norm_text = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", norm_text)
        norm_text = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", norm_text)

        if to_ascii:
            norm_text = ''.join(ch for ch in norm_text if unicodedata.category(ch)[0] != 'So')
            norm_text = unicodedata.normalize('NFKD', norm_text).encode('ascii', 'ignore').decode('utf-8')
        if to_lower:
            norm_text = norm_text.lower()

        while '""' in norm_text:
            norm_text = norm_text.replace('""', '"')
        norm_text = re.sub(r'\"+', '"', norm_text)
        norm_text = re.sub(r'\s+', ' ', norm_text).strip()
        return norm_text

    @staticmethod
    def normalize_series(text_series: pd.Series,
                         config=None,
                         to_ascii=True,
                         to_lower=False,
                         keep_emojis=True,
                         segment_hashtag=False,
                         username="@USER",
                         httpurl="HTTPURL") -> pd.Series:
        return text_series.apply(lambda txt: TextPreprocessor.normalize_text(txt,
                                                                             config=config,
                                                                             to_ascii=to_ascii,
                                                                             to_lower=to_lower,
                                                                             keep_emojis=keep_emojis,
                                                                             segment_hashtag=segment_hashtag,
                                                                             username=username,
                                                                             httpurl=httpurl))


if __name__ == "__main__":
    print(TextPreprocessor.normalize_text(
        "SC has first two presumptive cases of coronavirus, DHEC confirms "
        "https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms"
        "/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user"
        "-share… via @postandcourier #Covid_19"))
