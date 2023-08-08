import nltk # nltk - natural language toolkit
# nltk.download("punkt") 
# nltk.download("stopwords")
from nltk.corpus import stopwords #nltk.corpus provides us with an extensive list of stopwords to work with
from nltk.tokenize import word_tokenize 

example_sent = """This is a sample sentence,
                  showing off the stop words filtration."""

# creating a set of non repeating array of stopwords which can be refered to while removing them
stop_words = set(stopwords.words('english'))

# tokenizing each word in the given example set
word_tokens = word_tokenize(example_sent)

# using list comprehension for creating a list of stopword filtered array
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

print(f"Array tokenized words are as follows : {word_tokens}")
print(f"Array of stopword filtered sentence is as follows : {filtered_sentence}")