
import joblib
from gensim.models import Word2Vec
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np


nltk.download("stopwords")

stop_words = stopwords.words("russian")
df = pd.read_csv('example.csv')

segmenter = Segmenter()
morh_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)



def text_prep(text) -> str:
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        token.lemmatize(morh_vocab)

    lemmas = [_.lemma for _ in doc.tokens]
    words = [lemma for lemma in lemmas if lemma.isalpha() and len(lemma) > 2]
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

def train_model():

    df['content'] = df.content.apply(text_prep)

    model = Word2Vec(sentences=df.content.str.split(),
                 vector_size=200,
                 min_count=10,
                 window=2,
                 seed=32)

    joblib.dump(model, "model.pkl")


def load_model():
    return joblib.load('model.pkl')

def recommend_words(model, word, N):
    try:
        # Получаем N наиболее похожих слов
        similar_words = model.wv.most_similar(positive=[word], topn=N)
        #words = [item[0] for item in similar_word]
        # Формируем список рекомендаций в нужном формате
        recommendations = [
            {"entry_word": word, "words_predict": f"{similar_word}"}
            for i, (similar_word, _) in enumerate(similar_words)
        ]
        
        return recommendations
    
    except KeyError:
        # Если слово отсутствует в модели, выбрасываем исключение
        raise ValueError(f"Слово '{word}' отсутствует в базе данных.")

