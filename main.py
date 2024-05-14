import string
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import cm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Read the text file
text = open('read.txt', encoding ='utf-8').read()

# Convert text to lower case and remove punctuation
cleaned_text = text.lower().translate(str.maketrans('', '', string.punctuation))

# Tokenization
tokenized_words = word_tokenize(cleaned_text,"english")

# Remove stopwords
final_words = [word for word in tokenized_words if word not in stopwords.words('english')]

# Extract emotions from the text
emotion_list = []
with open('emotions.txt','r') as file:
    for line in file:
        clear_line = line.replace("\n",'').replace(",",'').replace("'",'').strip()
        word, emotion = clear_line.split(':')
        if word in final_words:
            emotion_list.append(emotion)

# Count the occurrences of each emotion
w_emotion = Counter(emotion_list)

# Sentiment analysis of the cleaned text
def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    if score['neg'] > score['pos']:
        print("Negative Sentiment")
    elif score['neg'] < score['pos']:
        print("Positive Sentiment")
    else:
        print("Neutral Sentiment")

# Plotting the emotions with enhanced styling
sorted_emotions = sorted(w_emotion.items(), key=lambda x: x[1], reverse=True)
emotions, counts = zip(*sorted_emotions)

plt.figure(figsize=(10, 6))

# Create a color gradient
colors = cm.plasma_r([i / len(emotions) for i in range(len(emotions))])

# Create bars with shadow effect and transparency
bars = plt.bar(emotions, counts, color=colors, alpha=0.8, edgecolor='grey', linewidth=1.5)

# Add counts on bars with custom font
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, count, ha='center', va='bottom',
             fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})

# Add labels and title with custom font
plt.xlabel('Emotions', fontsize=12, fontweight='bold', color='black')
plt.ylabel('Frequency', fontsize=12, fontweight='bold', color='black')
plt.title('Emotions in the Text', fontsize=14, fontweight='bold', color='black')

# Customize ticks and grid
plt.xticks(fontsize=10, fontweight='bold', color='black')
plt.yticks(fontsize=10, fontweight='bold', color='black')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show sentiment analysis
sentiment_analyse(cleaned_text)

# Save and display the plot
plt.tight_layout()
plt.savefig('emotions_graph.png', bbox_inches='tight', dpi=300, facecolor='lightgrey')
plt.show()