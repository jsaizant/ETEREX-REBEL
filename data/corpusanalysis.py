import nltk, re, string, numpy as np, statistics, os, os.path, nltk.data
from nltk import FreqDist, pos_tag
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.util import bigrams
from nltk.lm import MLE
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline, flatten, everygrams
from collections.abc import Iterable
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from matplotlib import pyplot as plt
from statistics import mean

# Load corpus
rootPath = os.path.expanduser('~/nltk_data/corpora/i2b2/')
corpus = PlaintextCorpusReader(rootPath, '.*\.txt', encoding = 'latin1')
corpuscos = []
dirlist = list(os.listdir(rootPath))
for i, file in enumerate(dirlist):
    if '.xml.txt' in file:
        #print(str(i)+' '+file)
        with open(file = rootPath+file, mode = 'rt') as f:
            corpuscos.append(' '.join(f.readlines()))

# Downloads
if not os.path.exists(os.path.expanduser('~/nltk_data/corpora/stopwords/')):
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('wordnet')

# Standard deviation function
def get_std_dev(lis):
    n = len(lis)
    mean = sum(lis) / n
    var = sum((x - mean)**2 for x in lis) / n
    std_dev = var ** 0.5
    return std_dev

# Flatten function
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:        
            yield item

# Corpus statistics
def stats(corpus):
    stats = []
    ##### Processing
    # Get sentence list from all documents
    doclist = [list(corpus.sents(fileids = fileid)) for fileid in corpus.fileids()]
    # Convert to lowercase
    doclist = [[[w.lower() for w in s] for s in doc] for doc in doclist]
    # Remove stopwords and punctuation and digits
    engstopwords = set(stopwords.words('english'))
    doclistfit = [[[w for w in s if w not in engstopwords and bool(re.search(r'\d', w))== False and bool(re.search(r'([^\w\s]|_)', w))== False] for s in doc] for doc in doclist]
    wordlistfit = list(flatten(doclistfit))
    ##### Measures
    ### Word document length
    doclenpopl = []
    for doc in doclistfit:
        doclen = 0
        for sent in doc:
            doclen += len(sent)
        doclenpopl.append(doclen)
    # Remove outliers
    qua = statistics.quantiles(doclenpopl, n=4)
    q1 = qua[0]
    q3 = qua[2]
    iqr = q3 - q1
    lowerbound = q1 - 1.5 * iqr
    upperbound = q3 + 1.5 * iqr
    doclenpopl = [doclen for doclen in doclenpopl if lowerbound <= doclen <= upperbound]
    # Average words per document
    avgwordsdoc = statistics.mean(doclenpopl)
    # Standard Deviation of document word length
    sdwordsdoc = statistics.stdev(doclenpopl)
    ### Word sentence length
    sentlenpopl = []
    for doc in doclistfit:
        for sent in doc:
            sentlen = len(sent)
            sentlenpopl.append(sentlen)
    # Remove outliers
    qua = statistics.quantiles(sentlenpopl, n=4)
    q1 = qua[0]
    q3 = qua[2]
    iqr = q3 - q1
    lowerbound = q1 - 1.5 * iqr
    upperbound = q3 + 1.5 * iqr
    sentlenpopl = [sentlen for sentlen in sentlenpopl if lowerbound <= sentlen <= upperbound]
    # Average words per sentence
    avgwordsent = statistics.mean(sentlenpopl)
    # Standard Deviation of document word length
    sdwordsent = statistics.stdev(sentlenpopl)
    ### Unique words
    uwordlist = list(set(wordlistfit))
    # Lexical density / Proportion of unique words in total words
    lexicaldensity = len(uwordlist) / len(wordlistfit)
    # Vocabulary / Frequency distribution of unique words
    vocabfreqdist = FreqDist(uwordlist)
    ### POS distribution
    postag = pos_tag(wordlistfit)
    tags = [tag for (_, tag) in postag]
    posfreqdist = FreqDist(tags)
    tags = np.unique(tags, return_counts = True)
    ### Compile stats
    stats.append('Number of documents: ' + str(len(corpus.fileids())))
    stats.append('Average document word length: ' + str(round(avgwordsdoc)))
    stats.append('Standard deviation of document word length: ' + str(round(sdwordsdoc)))
    stats.append('Average sentence word length: '+ str(round(avgwordsent)))
    stats.append('Standard deviation of sentence word length: ' + str(round(sdwordsent)))
    stats.append('Lexical density: ' + str(round(lexicaldensity, 2)))
    for stat in stats:
        print(stat)

#stats(corpus)

# Cosine similarity
def cos(corpus):
    ### Processing
    # Init
    lemmer = nltk.stem.WordNetLemmatizer()
    punctdigitdict = dict((ord(i), None) for i in string.punctuation + string.digits)
    engstopwords = set(stopwords.words('english'))
    # Tokenize function
    def LemTokens(tokens):
     return [lemmer.lemmatize(token) for token in tokens]
    # Lemmatize function
    def LemNormalize(text):
     return LemTokens(nltk.word_tokenize(text.lower().translate(punctdigitdict)))
    # Vectorizer
    LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words=engstopwords)
    # Create Term Frequency
    LemVectorizer.fit_transform(corpus)
    print(LemVectorizer.vocabulary_)
    # Create TF matrix
    tfmatrix = LemVectorizer.transform(corpus).toarray()
    print('TF Matrix shape: ' + str(tfmatrix.shape))
    # Calculate IDF
    tfidftrans = TfidfTransformer(norm="l2")
    tfidftrans.fit(tfmatrix)
    # Create TFIDF matrix
    tfidfmatrix = tfidftrans.transform(tfmatrix)
    # Pairwise similarity matrix
    cosmatrix = (tfidfmatrix * tfidfmatrix.T).toarray()
    # Average cosine similarity 
    print(round(np.mean(cosmatrix), 4))
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cosmatrix, interpolation='nearest', cmap = 'plasma')
    fig.colorbar(cax, location = 'top')
    plt.show()


#vectorizer = cos(corpuscos)

def perplexity(corpus):
    ### Processing
    # Get sentence list from all documents
    fileidlist = corpus.fileids()
    train, test = np.array_split(fileidlist, 2)
    def processing(corpus, idlist):
        sentlist = [corpus.sents(fileids = fileid) for fileid in idlist]
        sentlist = [s for doc in sentlist for s in doc]
        # Convert to lowercase
        sentlist = [[w.lower() for w in s] for s in sentlist]
        # Remove punctuation
        sentlist = [[w for w in s if w not in string.punctuation] for s in sentlist]
        return sentlist
    trainlist = processing(corpus, train)
    testlist = processing(corpus, test)
    # UNI
    unitrain, unitrainvocab = padded_everygram_pipeline(1, trainlist)
    unilm = MLE(1)
    unilm.fit(unitrain, unitrainvocab)
    print('uni')
    print(unilm.counts)
    unitest, _ = padded_everygram_pipeline(1, testlist)
    uniprplist = []
    for test in unitest:
        try:
            prp = unilm.perplexity(test)
            if prp != np.inf:
                uniprplist.append(prp)
        except ZeroDivisionError:
            print('err')
    print(round(mean(uniprplist), 2))
    

    # BI
    bitrain, bitrainvocab = padded_everygram_pipeline(2, trainlist)
    bilm = MLE(2)
    bilm.fit(bitrain, bitrainvocab)
    print('bi')
    print(bilm.counts)
    bitest, _ = padded_everygram_pipeline(2, testlist)
    biprplist = []
    for test in bitest:
        try:
            prp = bilm.perplexity(test)
            if prp != np.inf:
                biprplist.append(prp)
        except ZeroDivisionError:
            print('err')
    print(round(mean(biprplist), 2))

perplexity(corpus)
