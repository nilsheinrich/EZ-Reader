from EZReader.Corpus import Corpus
from EZReader.Model import run_EZReader

# fixed simulation parameters:
maxLength = 20  # maximum of letters + space left of word
maxSentenceLength = 50  # maximum amount of words in sentence

corpus_file = open("SRC98Corpus.txt", "r")

Symbol = "@"
text = corpus_file.read().replace('\n', ' ')
NSentences = text.count(Symbol)

text_array = {}  # open dictionary in which sentences are appended later as objects
corpus = Corpus()
corpus.initialize(maxLength, text_array)

results = run_EZReader(text_array, corpus, NSentences, maxSentenceLength)

print(results)
