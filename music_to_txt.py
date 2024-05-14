import os
import speech_recognition as sr
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import cm


def record_audio_to_file(file_path, min_duration=5):  # Setting the minimum duration to 5 seconds
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak...")
        recognizer.adjust_for_ambient_noise(source)

        audio = recognizer.listen(source,timeout=min_duration)  # Setting the timeout to record audio for at least min_duration seconds

    # Save audio to file
    with open(file_path, "wb") as f:
        f.write(audio.get_wav_data())

    # Calculate the actual duration of the recorded audio
    audio_duration = len(audio.frame_data) / (audio.sample_rate * audio.sample_width)
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print("Audio saved to:", file_path)


def transcribe_audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API: {e}")
        return None


def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    compound_score = sentiment_score['compound']

    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# Specify the path to save the audio file
audio_file_path = "recorded_audio.wav"

# Record audio (minimum 5 seconds) and save to file
record_audio_to_file(audio_file_path, min_duration=5)

# Transcribe audio to text
transcribed_text = transcribe_audio_to_text(audio_file_path)

if transcribed_text:
    print("Transcribed text:", transcribed_text)

    # Analyze sentiment
    sentiment = analyze_sentiment(transcribed_text)
    print("Sentiment:", sentiment)

    # Count emotions
    sia = SentimentIntensityAnalyzer()
    sentences = transcribed_text.split('.')
    emotions = []
    for sentence in sentences:
        sentiment_score = sia.polarity_scores(sentence)
        if sentiment_score['compound'] >= 0.05:
            emotions.append("Positive")
        elif sentiment_score['compound'] <= -0.05:
            emotions.append("Negative")
        else:
            emotions.append("Neutral")

    emotion_counts = Counter(emotions)

    sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
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

    # Save and display the plot
    plt.tight_layout()
    plt.savefig('emotions_graph.png', bbox_inches='tight', dpi=300, facecolor='lightgrey')
    plt.show()
else:
    print("No text transcribed from the audio.")
