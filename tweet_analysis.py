import pandas as pd
import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Read tweets from CSV file with a limit
limit = 100
tweets_df = pd.read_csv('Twitter_Data.csv', nrows=limit)

# Assuming your CSV has a column named 'tweet' containing the tweets
tweets = tweets_df['tweet'].tolist()

# Combine tweets into a single text
text = " ".join(tweets)

# Convert to lowercase
lower_case = text.lower()

# Remove punctuations
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

# Tokenize text
tokenized_words = word_tokenize(cleaned_text, "english")

# Remove stopwords
final_words = [word for word in tokenized_words if word not in stopwords.words('english')]

# Extract emotions from the text
emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')
        if word in final_words:
            emotion_list.append(emotion)

# Count emotions
emotion_counts = Counter(emotion_list)

def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    if score['compound'] >= 0.05:
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Plot emotions with enhanced visualizations
fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # Create subplots

# Plot bar chart for emotion frequency
colors = plt.cm.Paired(range(len(emotion_counts)))  # Color scheme
bars = axs[0].bar(emotion_counts.keys(), emotion_counts.values(), color=colors)

# Add shadow effect to bars
for bar in bars:
    bar.set_edgecolor('grey')
    bar.set_linewidth(0.8)
    bar.set_alpha(0.7)

# Add labels and title to bar chart
axs[0].set_xlabel('Emotions', fontsize=12, fontweight='bold')
axs[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axs[0].set_title('Sentiment Analysis of Tweets', fontsize=14, fontweight='bold')
axs[0].grid(axis='y', linestyle='--', alpha=0.7)
axs[0].set_facecolor('#f0f0f0')
plt.sca(axs[0])  # Set current axis

# Rotate x-axis labels for better readability
plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

# Plot sentiment distribution pie chart
sentiment_labels = ['Positive', 'Negative', 'Neutral']
sentiment_distribution = [0, 0, 0]
for tweet in tweets:
    sentiment = sentiment_analyse(tweet)
    if sentiment == 'Positive':
        sentiment_distribution[0] += 1
    elif sentiment == 'Negative':
        sentiment_distribution[1] += 1
    else:
        sentiment_distribution[2] += 1

# Create an exploded pie chart
explode = (0.1, 0, 0)  # Explode the positive slice
colors_pie = ['lightgreen', 'lightcoral', 'lightskyblue']
shadow_pie = True  # Add shadow effect

# Calculate the size of the pie chart relative to the bar chart
ratio = 0.25
width_ratio = 1 - ratio

# Set the width ratio of the subplots
fig.subplots_adjust(wspace=width_ratio)

# Plot the pie chart
axs[1].pie(sentiment_distribution, labels=sentiment_labels, autopct='%1.1f%%', startangle=140, explode=explode,
            colors=colors_pie, shadow=shadow_pie)
axs[1].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
axs[1].legend(loc="best", fontsize=10)  # Add legend

# Adjust layout
plt.tight_layout()

plt.show()
