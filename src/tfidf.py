import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk.corpus import stopwords

with open("tvqarm.txt") as fp:
    stopwords=fp.read()
stopwords=nltk.word_tokenize(stopwords)


def tfidf(sentence):
    corpus = [sentence]
    vectorizer = CountVectorizer(stop_words=stopwords)
    transformer = TfidfTransformer()
    X = vectorizer.fit_transform(corpus)
    word = vectorizer.get_feature_names()

    with open("feature.pkl","wb") as fw:
        pickle.dump(vectorizer.vocabulary_, fw)
    with open("transformer.pkl", 'wb') as fw:
        pickle.dump(transformer, fw)

    Tfidf = transformer.fit_transform(X)
    Tfidf=Tfidf.toarray()
    print("Tfidf="+str(Tfidf))
    print("len(Tfidf[0])="+str(len(Tfidf[0])))
    return Tfidf

with open("tvqaall.txt") as fp1:
    loadir1=fp1.read()
    fullsentence=loadir1
    tfidf(fullsentence)