import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import words

nltk.download('punkt')
nltk.download('words')

# Input paragraph
paragraph = "This is a sample paragraph. It contains multiple sentences with special characters like @#$%. It also has spellinng mistaakes."

# Tokenize the paragraph into sentences
sentences = sent_tokenize(paragraph)

# Initialize the English words set for spelling correction
english_words = set(words.words())

# List to store cleaned and corrected sentences
cleaned_sentences = []

# Function to remove special characters, correct spellings, and remove extra spaces
def clean_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Correct spellings
    corrected_words = [word if word.lower() in english_words else nltk.corpus.words.correct(word) for word in word_tokenize(text)]
    corrected_text = ' '.join(corrected_words)
    
    # Remove extra spaces
    cleaned_text = ' '.join(corrected_text.split())
    
    return cleaned_text

# Clean and append sentences to the cleaned_sentences list
for sentence in sentences:
    cleaned_sentence = clean_text(sentence)
    cleaned_sentences.append(cleaned_sentence)

# Display the cleaned sentences
for sentence in cleaned_sentences:
    print(sentence)
