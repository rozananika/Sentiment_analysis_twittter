from flask import Flask, render_template, flash
import tweepy
import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np
import matplotlib.pyplot as plt 
import io
import base64
from wordcloud import WordCloud, STOPWORDS
import json
import time

app = Flask(__name__)
app.secret_key = '***'  # Required for flashing messages

def analyze_tweets(query='#InspiringWomen'):
    try:
        # Twitter Authentication
        client = tweepy.Client(bearer_token='****')
        
        # Pull tweets with rate limit handling
        tweet_lists = []
        try:
            paginator = tweepy.Paginator(client.search_recent_tweets, 
                                       query=query, 
                                       max_results=10,
                                       limit=2)
            
            for tweet in paginator.flatten(limit=20):
                tweet_lists.append(tweet)
                time.sleep(1)
                
        except tweepy.TooManyRequests:
            if not tweet_lists:
                return {
                    'tweets': [],
                    'sentiment_data': [0, 0, 0],
                    'positive_wordcloud': "",
                    'negative_wordcloud': "",
                    'error': "Twitter API rate limit reached. Please try again later."
                }
        
        if not tweet_lists:
            return {
                'tweets': [],
                'sentiment_data': [0, 0, 0],
                'positive_wordcloud': "",
                'negative_wordcloud': "",
                'error': "No tweets found for the given query."
            }
        
        # Create DataFrame
        tweet_lists_df = pd.DataFrame(tweet_lists)
        tweet_lists_df = pd.DataFrame(tweet_lists_df[['text']])
        
        # Preprocess tweets
        def preprocess_tweet(sen):
            sentence = sen.lower()
            sentence = re.sub(r'RT @\w+:', '', sentence)
            sentence = re.sub(r"(@[A-Za-z0-9]+)|(^0-9A-Za-z \t)|(\w+:\/\/\S+)", "", sentence)
            sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
            sentence = re.sub(r'\s+', ' ', sentence)
            return sentence
        
        cleaned_tweets = []
        for tweet in tweet_lists_df['text']:
            cleaned_tweet = preprocess_tweet(tweet)
            cleaned_tweets.append(cleaned_tweet)
        
        tweet_lists_df['cleaned_tweets'] = cleaned_tweets
        
        # Generate sentiment
        tweet_lists_df[['polarity', 'subjectivity']] = tweet_lists_df['cleaned_tweets'].apply(
            lambda Text: pd.Series(TextBlob(Text).sentiment))
        
        for index, row in tweet_lists_df['cleaned_tweets'].items():
            score = SentimentIntensityAnalyzer().polarity_scores(row)
            neg = score['neg']
            neu = score['neu']
            pos = score['pos']
            comp = score['compound']
            
            if comp <= -0.05:
                tweet_lists_df.loc[index, 'sentiment'] = 'Negative'
            elif comp >= 0.05:
                tweet_lists_df.loc[index, 'sentiment'] = 'Positive'
            else:
                tweet_lists_df.loc[index, 'sentiment'] = 'Neutral'
                
            tweet_lists_df.loc[index, 'neg'] = neg
            tweet_lists_df.loc[index, 'neu'] = neu
            tweet_lists_df.loc[index, 'pos'] = pos
            tweet_lists_df.loc[index, 'compound'] = comp
        
        # Generate sentiment counts with default values
        sentiment_counts = pd.Series([0, 0, 0], index=['Positive', 'Negative', 'Neutral'])
        actual_counts = tweet_lists_df['sentiment'].value_counts()
        sentiment_counts.update(actual_counts)
        
        # Generate word clouds
        positive_tweets = tweet_lists_df['cleaned_tweets'][tweet_lists_df["sentiment"] == 'Positive']
        negative_tweets = tweet_lists_df['cleaned_tweets'][tweet_lists_df["sentiment"] == 'Negative']
        
        # Create word clouds
        stop_words = ["https", "co", "RT"] + list(STOPWORDS)
        
        def generate_wordcloud(text):
            try:
                if not text.empty:
                    wordcloud = WordCloud(max_font_size=50, max_words=50, 
                                    background_color="white", 
                                    stopwords=stop_words).generate(str(text))
                    
                    img = io.BytesIO()
                    plt.figure(figsize=(10,5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.savefig(img, format='png', bbox_inches='tight')
                    plt.close()
                    img.seek(0)
                    return base64.b64encode(img.getvalue()).decode()
                return ""
            except Exception as e:
                print(f"Error generating wordcloud: {e}")
                return ""
        
        positive_wordcloud_img = generate_wordcloud(positive_tweets)
        negative_wordcloud_img = generate_wordcloud(negative_tweets)
        
        return {
            'tweets': tweet_lists_df.to_dict('records'),
            'sentiment_data': sentiment_counts.values.tolist(),
            'positive_wordcloud': positive_wordcloud_img,
            'negative_wordcloud': negative_wordcloud_img,
            'error': None
        }
    except Exception as e:
        return {
            'tweets': [],
            'sentiment_data': [0, 0, 0],
            'positive_wordcloud': "",
            'negative_wordcloud': "",
            'error': str(e)
        }

@app.route('/')
def index():
    analysis_results = analyze_tweets()
    if analysis_results.get('error'):
        flash(analysis_results['error'], 'error')
    return render_template('index.html', 
                         tweets=analysis_results['tweets'],
                         sentiment_data=analysis_results['sentiment_data'],
                         positive_wordcloud=analysis_results['positive_wordcloud'],
                         negative_wordcloud=analysis_results['negative_wordcloud'])

if __name__ == '__main__':
    nltk.download('vader_lexicon')
    app.run(debug=True)
