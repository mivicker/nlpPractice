import nltk
from nltk.corpus import wordnet as wn

vehicle = wn.synsets('vehicle')[0]

t = nltk.Tree(vehicle.name(), children = [
    nltk.Tree(vehicle.hyponyms()[3].name(), children=[]),
    nltk.Tree(vehicle.hyponyms()[4].name(), children=[]),
    nltk.Tree(vehicle.hyponyms()[5].name(), children=[]),
    nltk.Tree(vehicle.hyponyms()[6].name(), children=[
        nltk.Tree(vehicle.hyponyms()[7].name(), children=[]),
        nltk.Tree(vehicle.hyponyms()[7].name(), children=[]),
        nltk.Tree(vehicle.hyponyms()[7].name(), children=[]),
        nltk.Tree(vehicle.hyponyms()[7].name(), children=[])
])
])

t.draw()
