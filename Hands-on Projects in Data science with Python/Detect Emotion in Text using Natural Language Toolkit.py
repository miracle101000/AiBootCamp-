# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from  nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Download required NLTK data (only needed once)
nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initializer the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Sample text for emotion detection
text = """

"""

# Function to detect emotion in text
def detect_emotion(text):
    # Analyze sentiment
    scores = sid.polarity_scores(text)
    
    # Display sentiment scores
    print("Sentiment Scores:", scores)
    
    # Determine emotion based on scores
    if scores["compound"] >= 0.5:
        emotion = "Joy"
    elif scores["compound"] <= -.5:
        emotion = "Sadness"
    elif scores["neg"] > .5:
        emotion = "Anger"
    elif scores["neu"] >.7:
        emotion = "Neutral"    
    else:
        emotion = "Mixed emotions"            
    
    return emotion    

# Detect and print the emotion
emotion = detect_emotion(text)
print("Detected Emotion:", emotion)