
# Sentiment Analysis (TWEETS AND MUSIC)
This project contains Python scripts for analyzing sentiment in text data and processing audio files. It includes functionalities to record audio, transcribe it to text, and perform sentiment analysis on the transcribed text.

 Overview
The project consists of four main Python scripts:

 Main.py: 
This script analyzes the sentiment of text data stored in a file named read.txt. It cleans the text, tokenizes it, removes stopwords, and identifies emotions present in the text. The sentiment analysis results are visualized using a bar plot.

 Tweets Analysis.py:
This script reads tweet data from a CSV file named Twitter_Data.csv, performs sentiment analysis on the tweets, and generates visualizations of sentiment distribution using bar charts and pie charts.

 Music.py:
This script records audio from the microphone, transcribes it to text using Google Web Speech API, performs sentiment analysis on the transcribed text, and visualizes emotion distribution using a bar plot.

 Setting.py:  
This script downloads necessary NLTK resources for text processing, such as tokenizers and sentiment lexicons.

Usage
Main.py:

Store the text data to be analyzed in a file named read.txt.
Run the script using python main.py.
Tweets Analysis.py:

Prepare a CSV file named Twitter_Data.csv containing tweet data with a column named tweet.
Run the script using python tweets_analysis.py.
Music.py:

Run the script to record audio and analyze sentiment using python music.py.
Setting.py:

Run the script to download necessary NLTK resources using python setting.py.
Dependencies
 NLTK: Natural Language Toolkit for text processing
 Matplotlib: Python plotting library for data visualization
 Pandas: Data manipulation and analysis library
 SpeechRecognition: Library for performing speech recognition
Additional Files
 Emotion.txt: Contains a list of words associated with specific emotions, used as a reference for identifying emotions in text data.


This README provides a brief overview of the project, its functionalities, usage instructions, dependencies, and additional files. Feel free to customize it further based on your project's specific requirements and details.
