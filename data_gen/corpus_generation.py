from nltk.corpus import product_reviews_1
import string
import pickle


camera_reviews = product_reviews_1.reviews('Canon_G3.txt')
review = camera_reviews[0]
rev = [map(str,review.sents()[i]) for i in xrange(len(review.sents()))]
for k in xrange(len(rev)):
    rev_tmp = [''.join(j for j in i if j not in string.punctuation) for i in rev[k]]
    rev[k] = [s for s in rev_tmp if s]
pickle.dump(rev,open('../data/Corpus.pkl','w'))