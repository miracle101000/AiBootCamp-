import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download stopwords (only needed once)
nltk.download("stopwords")
nltk.download("punkt")

# Example text for summarization 
text = """

"""

# Function to generate a frequency based summary
def summarize_text(text, num_senetences=2):
    # Tokenize text intoo sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # Filter out stopwords and non-alphabetic words
    stop_words = set(stopwords.words("english"))
    word_frequencies = {}
    
    for word in words:
        if word.isalpha() and word not in stop_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1
    
    # Score each sentences based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies
    
    # Sort sentences by score and select the top `num_sentences`
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_senetences]            
    summary = " ".join(summary_sentences)
    return summary

# Generate and print the summary
summary = summarize_text(text, num_senetences=2)
print("Original Text:\n", text)
print("\Summary:\n", summary)