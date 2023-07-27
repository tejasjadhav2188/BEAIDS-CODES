import nltk # nltk - natural language toolkit
# nltk.download("punkt") 
# nltk.download("stopwords")
from nltk.corpus import stopwords #nltk.corpus provides us with an extensive list of stopwords to work with
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
from functools import reduce

ps = PorterStemmer() 

example_sent = """Moin is running and running and running."""
 
# creating a set of non repeating array of stopwords which can be refered to while removing them
stop_words = set(stopwords.words('english'))
 
# tokenizing each word in the given example set
word_tokens = word_tokenize(example_sent)

# using list comprehension for creating a list of stopword filtered array
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

# using reduce and lambda function for creating a list of stemmed words
stemmed_sentence = reduce(lambda x, y: x + " " + ps.stem(y), word_tokens, "") 

print(f"Array tokenized words : {word_tokens}")
print(f"Array of stopword filtered sentence : {filtered_sentence}")
print(f"Array of stemmed words : {stemmed_sentence}")