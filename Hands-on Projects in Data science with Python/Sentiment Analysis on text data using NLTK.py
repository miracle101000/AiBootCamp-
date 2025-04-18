# Import necessary libraries
import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.corpus import stopwords
import random

# Download the NLTK data files
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess the dataset and extract features
def extract_features(words):
    return {word: True for word in words}

# Load the movie_reviews datset from NLTK
documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]    

# Shuffle the dataset to ensure random distribution
random.shuffle(documents)

# Prepare the dataset for training and testing
fetauresets = [(extract_features(d), c) for (d,c) in documents]
train_set, test_set = fetauresets[:1600], fetauresets[1600:]

# Train the Naive Bayes Classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate the classifier on the test set
accuracy = nltk_accuracy(classifier, test_set)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Show the most informative features
classifier.show_most_informative_features(10)

# Test on new input sentences
def analyze_sentiment(text):
    # Tokenize and remove stopwords
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words('english')]
    
    # Predict sentiment
    features = extract_features(words)
    return classifier.classify(features)

# Test the classifier with some custom text inputs
test_sentences =[
    "This movie is absolutely fantastic! The acting, the story, everyhtoing was amazing",
    "I hated this movie. It was waste of time and money.",
    "The plot was a bit dull, but the perfomances were great.",
    "I have mixed feelings about this film. It was okay, no great but not terrible either"
]

for sentence in test_sentences:
    print(f"Sentence: {sentence}")
    print(f"Predicted sentiment: {analyze_sentiment(sentence)}")
        
