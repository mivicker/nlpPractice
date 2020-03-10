#Continuing to work from NLP for hackers.
import nltk
from nltk import word_tokenize
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder

#Load ulysees into a variable.
with open('messages_only.txt', 'r',  encoding="utf-8") as myfile:
    text = myfile.read()

#tokenize the text
tokens = word_tokenize(text)

bigram_measures = BigramAssocMeasures()
trigram_measures = TrigramAssocMeasures()

#compute length-2 collocations
finder = BigramCollocationFinder.from_words(tokens)

finder.apply_freq_filter(5)

print(finder.nbest(bigram_measures.pmi, 20))

finder = nltk.collocations.TrigramCollocationFinder.from_words(tokens)

#only trigrams that appear 5+ times
finder.apply_freq_filter(5)

#return the 50 trigrams with the highest PMI
print(finder.nbest(trigram_measures.pmi, 20))
