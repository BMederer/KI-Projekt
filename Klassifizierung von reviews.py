#
# Projekt: KI-gestützte Klassifikation von Kundenbewertungen
# Ziel: Texte analysieren, Sentiment & Thema bestimmen
# Anmerkung: Dummy-KI, um API-Quota zu umgehen
#

import pandas as pd
#import openai
#import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\Berni\PycharmProjects\Erstes Projekt\KI-Projekt\reviews_german_realistic_utf8.csv')

# Überblick über die Daten
print(df.head())
print(df.describe())

# kleine Auswertung der Daten
# Wie viele Bewertungen pro Stern?
print(df['star_rating'].value_counts().sort_values(ascending=True))

# Durchschnittliche hilfreiche Stimmen
print('Durchschnittliche hilfreiche Stimmen', df['helpful_votes'].mean())

# Länge der Reviews
df['text_length'] = df['review_text'].apply(len)
print(df['text_length'].describe())

#openai.api_key = ''


'''def classify_review(text):
    prompt = f"""
    Analysiere diese deutsche Bewertung und antworte nur mit zwei Wörtern, getrennt durch ein Komma:
    1. Sentiment: positiv, neutral oder negativ
    2. Thema: Preis, Qualität, Lieferung oder anderes
    Bewertung: "{text}"
    Antwortformat: sentiment, topic
    """

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result = response.choices[0].message.content.strip()

    if "," in result:
        sentiment, topic = result.split(",")
        return sentiment.strip(), topic.strip()
    else:
        return "neutral", "anderes"
'''

def classify_review_dummy(text):
    if any(word in text.lower() for word in ["gut", "super", "toll"]):
        return "positiv", "Qualität"
    elif any(word in text.lower() for word in ["schlecht", "kaputt"]):
        return "negativ", "Qualität"
    else:
        return "neutral", "anderes"

sentiments = []
topics = []

for text in df['review_text']:
    s, t = classify_review_dummy(text)
    sentiments.append(s)
    topics.append(t)

df['sentiment'] = sentiments
df['topic'] = topics

print(df['sentiment'].value_counts())
print(df['topic'].value_counts())

