# -*- coding: utf-8 -*-
#/usr/bin/python2

import re
import nltk.data

# Returns formatted paragraph text
# [[p1Word1, p1Word2, ...], [p2Word1, p2Word2, ...]]
def getParagraphs(book, chapter):
    textPath = "books/book%d/ch%d.txt" % (book, chapter)

    with open(textPath, 'r', encoding="utf-8") as textFile:
        text = textFile.read()
        text = text.replace('“','"').replace('”','"')
        text = text.replace("’","'")

    paragraphs = re.split(r'(\n)', text)
    paragraphs = [p.split() for p in paragraphs]

    # Filter the paragraphs to make sure that we're getting reasonable
    # content (not newlines, empty characters, etc)
    filteredParagraphs = []
    for p in paragraphs:
        paragraph = []
        for w in p:
            if w == '' or w == '\n': continue
            paragraph.append ( w )
        if len(paragraph) == 0: continue
        filteredParagraphs.append(paragraph)
    return filteredParagraphs

def getSentences(book, chapter):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    textPath = "books/book%d/ch%d.txt" % (book, chapter)
    with open(textPath, 'r', encoding="utf-8") as textFile:
        text = textFile.read()
        text = text.replace('“','"').replace('”','"')
        text = text.replace("’","'")

    sentencesStrings = tokenizer.tokenize(text)
    sentences = []

    # Convert the sentence strings into words
    for s in sentencesStrings:
        sentence = s.split()
        sentences.append(sentence)

    return sentences
