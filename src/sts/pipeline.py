from sts.utils.preprocessor import TextPreprocessor
from sts.utils.embedder import TransformersEmbedder
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.preprocessing import normalize
from dotenv import load_dotenv, find_dotenv
import json
import os
import pandas as pd
import torch
import random
import numpy as np


class SemanticTextualSimilarityPipeline:

    def __init__(self, config_dirpath=None, stored_texts: pd.Series = None):
        load_dotenv(find_dotenv())
        if config_dirpath is None:
            config_dirpath = os.environ['STS_CONFIG_DIRPATH']
        with open(os.path.join(config_dirpath, 'config.json')) as JSON:
            self._main_config = json.loads(JSON.read())
        selected_pretrained_model = self._main_config['selected_pretrained_model']
        self._device = self._main_config['device']
        self._random_state = self._main_config['random_state']
        if self._device == 'default':
            self._device = torch.device(os.environ['DEFAULT_DEVICE'])
        else:
            self._device = torch.device(self._device)

        pretrained_config_path = os.path.join(config_dirpath, 'pretrained_config')
        pretrained_config_path = os.path.join(pretrained_config_path, selected_pretrained_model)
        with open(os.path.join(pretrained_config_path, 'preprocessing.json')) as JSON:
            self._preprocessor_config = json.loads(JSON.read())

        with open(os.path.join(pretrained_config_path, 'tokenizer.json')) as JSON:
            self._encode_config = json.loads(JSON.read())
            self._tokenizer = AutoTokenizer.from_pretrained(selected_pretrained_model)

        with open(os.path.join(pretrained_config_path, 'model.json')) as JSON:
            random.seed(self._random_state)
            np.random.seed(self._random_state)
            torch.manual_seed(self._random_state)
            torch.cuda.manual_seed_all(self._random_state)

            self._model_config = AutoConfig.from_pretrained(selected_pretrained_model,
                                                            **json.loads(JSON.read()))
            self._model = AutoModel.from_pretrained(selected_pretrained_model,
                                                    config=self._model_config)
            self._model.to(self._device)

        self.stored_texts = stored_texts
        if stored_texts is not None:
            self.stored_norm_text_embeddings = normalize(self._embed_series(stored_texts),
                                                         norm=self._main_config['embedding_norm'])

    def _preprocess_text(self, text: str):
        return TextPreprocessor.normalize_text(text, config=self._preprocessor_config)

    def _embed_text(self, text: str):
        text = self._preprocess_text(text)
        embedding_type = self._main_config['embedding_type']
        assert embedding_type in ['text-vector1d', 'last-layer-features']
        if embedding_type == 'text-vector1d':
            return TransformersEmbedder.text_vector(text=text,
                                                    pretrained_model=self._model,
                                                    tokenizer=self._tokenizer,
                                                    encode_config=self._encode_config,
                                                    device=self._device,
                                                    return_tensors=False,
                                                    random_state=self._random_state)
        else:
            return TransformersEmbedder.last_layer_features(text=text,
                                                            pretrained_model=self._model,
                                                            tokenizer=self._tokenizer,
                                                            encode_config=self._encode_config,
                                                            device=self._device,
                                                            return_tensors=False,
                                                            random_state=self._random_state)

    def _preprocess_series(self, text_series: pd.Series):
        return np.vectorize(lambda txt: self._preprocess_text(txt))(text_series)

    def _embed_series(self, text_series: pd.Series):
        embedded = []
        for txt in text_series:
            embedded.append(self._embed_text(txt).reshape(1, -1))
        embedded = np.vstack(embedded)
        return embedded

    def pairwise_cossim(self, text_a: str, text_b: str):
        norm = self._main_config['embedding_norm']
        v1 = self._embed_text(text_a)
        v2 = self._embed_text(text_b)
        return (normalize(v1.reshape(1, -1), norm=norm)
                @ normalize(v2.reshape(1, -1), norm=norm).T).item()

    def set_stored_texts(self, stored_texts: pd.Series):
        self.stored_norm_text_embeddings = normalize(self._embed_series(stored_texts),
                                                     norm=self._main_config['embedding_norm'])

    def get_stored_best_matches(self,
                                input_text: str,
                                nbest=3,
                                return_indices=False):
        norm = self._main_config['embedding_norm']
        input_embedding = normalize(self._embed_text(input_text).reshape(1, -1), norm=norm)
        scores = (self.stored_norm_text_embeddings @ input_embedding.T).squeeze()
        indices = np.argsort(scores)
        indices = indices[-nbest:][::-1]
        if return_indices:
            return indices, scores[indices]
        return self.stored_texts[indices].to_numpy(), scores[indices]

    def get_stored_best_match(self, input_text: str, return_indices=False):
        result, score = self.get_stored_best_matches(input_text,
                                                     nbest=1,
                                                     return_indices=return_indices)
        return result.item(), score.item()

    def get_best_matches(self, input_text: str, ref_texts: pd.Series, nbest=1, return_indices=False):
        norm = self._main_config['embedding_norm']
        input_embedding = normalize(self._embed_text(input_text).reshape(1, -1), norm=norm)
        ref_embeddings = normalize(self._embed_series(ref_texts), norm=norm)
        scores = (ref_embeddings @ input_embedding.T).squeeze()
        indices = np.argsort(scores)[-nbest:][::-1]
        if return_indices:
            return indices, scores[indices]
        return ref_texts[indices].to_numpy(), scores[indices]
