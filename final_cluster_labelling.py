import urllib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.stem
import operator
import wikipedia
import collections
from nltk.util import bigrams
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import load_files
import collections as C
from nltk.collocations import *
trigram_measures = nltk.collocations.TrigramAssocMeasures()
from nltk.util import trigrams
from nltk.util import ngrams
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import re
import pickle



##############
# clustering #
##############

dataset_files = load_files("./20_newsgroups.parent")

#number of clusters 
true_k = 1

f = open("dataset_files_output.txt","wb")
f.write(str(dataset_files))
f.close()


vectorizer = TfidfVectorizer(stop_words='english',decode_error='ignore')    #decode error ignore cuz newsgroups contained data with \u, etc characters.
X = vectorizer.fit_transform(dataset_files.data)
model = KMeans(n_clusters=true_k, init='k-means++', n_init=50, max_iter=1000)
model.fit(X)

clusters = C.defaultdict(list)

print "length is ",len(dataset_files.filenames)
k = 0
for i in model.labels_:
    clusters[i].append(dataset_files.filenames[k])  
    k += 1

#########################################################################
# making a list of list, where each list represents data of one cluster
#########################################################################


list_of_lists = []
for clust in clusters:
    print "\n***********************************\n"
    print "clust = ",clust
    list_of_data_in_cluster = []
    #print "length of this cluster : ",len(clust)
    for i in clusters[clust]:
        print i
        file_open = open(i)
        list_of_data_in_cluster.append(file_open.read())
        file_open.close()
    list_of_lists.append(list_of_data_in_cluster)
        


################################################################################
#opening files in one cluster and putting that in docs and working on it
################################################################################


#docs = ["information systems","computer relevance","restart boot","core 2 duo networks software computer","logic system"]
# print "docs array made."

for x in range(0,len(list_of_lists)):               #for each cluster, do : 
    
    print "=============== x = ",x," ================"              #will denote cluster number
    docs = list_of_lists[x]                                         
    fw = open('corpus','w')
    
    fw.write((','.join(docs)).replace(","," "))                     #writing everything of one cluster in a file.

    countDocuments = len(docs)                                      #number of documents in this cluster

    
    ####################################################################################################################################################
    # Extending Tfidf to have only stemmed features - IMPORTANT : Not using stemming because it causes words to become unrecognizably small. Eg : ae, si
    ####################################################################################################################################################
    english_stemmer = nltk.stem.SnowballStemmer('english')

    class StemmedTfidfVectorizer(TfidfVectorizer):
        def build_analyzer(self):
            analyzer = super(TfidfVectorizer, self).build_analyzer()
            #print "Analyzer"+analyzer+"\n\n\n"
            # return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))    #FOR STEMMING, UNCOMMENT THIS. 
            return lambda doc: (w for w in analyzer(doc))

    tfidf = StemmedTfidfVectorizer(stop_words='english', decode_error='ignore', ngram_range=(1,1))  

    print "stopwords removal done."

    Xv = tfidf.fit_transform(docs) 

    print "Xv (tf-idf) matrix done."

    ##################################################################
    # Writing the keywords, whole text, and the tf-idf matrix in file
    ##################################################################

    feature_names = tfidf.get_feature_names()



    f = open("output-dummyIndexing.txt",'w')
    f.write(str(feature_names))
    f.write("\n")
    f.write("\n")
    f.write("\n")

    f.write(str(docs))
    f.write("\n")
    f.write("\n")
    f.write("\n")
    f.write(str(Xv))
    f.write("\n")
    f.write("\n")
    f.write("\n")


    ##########################################
    # FINDING Inverted Indexing as we need it
    ##########################################

    inverseIndexingDict = {}
    #print "num of features",len(feature_names) 

    for j in range(0,len(feature_names)):
        inverseIndexingDict[feature_names[j]] = []
        for i in range(0,countDocuments):
            inverseIndexingDict[feature_names[j]].append(Xv[i,j])

    file_inverseIndex = "inverseIndexingDict"+str(x)+".txt"
    g = open(file_inverseIndex,"w")
    g.write(str(inverseIndexingDict))
    g.close()

    print "inverseIndexingDict done"
    print "len(inverseIndexingDict) = ",len(inverseIndexingDict)

    newInverseIndexingDict = {}
    ########################################################################
    # CHOOSING only those terms which are present in more than 3 documents
    # 3 was experimentally observed to have worked well.
    ########################################################################
    for j in range(0,len(feature_names)):
        a = inverseIndexingDict[feature_names[j]]       # a is a list of tf-idf values for a particular keyword with all docs
        count_nonzero = 0
        for item in a:
            if item != 0:
                count_nonzero += 1
        if count_nonzero > 3:
            newInverseIndexingDict[feature_names[j]] = inverseIndexingDict[feature_names[j]]

    #writing new inverse dictionary to a file.
    print "newInverseIndexingDict done "
    print "len(newInverseIndexingDict) = ",len(newInverseIndexingDict)
    filename = "newInverseIndexingDict"+str(x)+".txt"
    file1 = open(filename,"wb")
    file1.write("Keys(keywords) are : \n\n\n")
    file1.write(str(newInverseIndexingDict.keys()))
    file1.write("\n\n\n\n")
    file1.write("Whole dictionary is : \n\n")
    file1.write(str(newInverseIndexingDict))
    file1.close() 

    #####################################################################
    # Finding centroid of all keywords using inverse indexing dictionary
    #####################################################################
    centroid = []
    for i in range(0,countDocuments):
        sum = 0
        for key in newInverseIndexingDict.keys():
            sum += newInverseIndexingDict[key][i]
        centroid.insert(i,sum/len(feature_names))
        # centroid.insert(i,0)                  # in case you want centroid to be origin.
    
    #print str(centroid)
    print "Done with matrix and inverted indexing and centroid."


    ############# inverseIndexingDict details ################
    ### keys are terms
    ### values is a list of tfidf with each document
    ##########################################################

    #####################################################################################
    # Finding distance of every term from centroid and then finding farthest ones
    #####################################################################################

    

    termCentroidDistDict = {}
    for term in newInverseIndexingDict.keys():
        dist = 0
        for i in range(0,countDocuments):
            dist += (centroid[i]-newInverseIndexingDict[term][i])**2
        termCentroidDistDict[term] = math.sqrt(dist)            #LAST-MINUTE

    #print "The term to centroid distance is"
    #print termCentroidDistDict      #key : term, value : euclidean distance from centroid (considering term as a vector

    #sorted representation - list of tuples.
    sorted_centroid_dist = sorted(termCentroidDistDict.items(), key=operator.itemgetter(1), reverse=True)

    #writing it to a file.
    filename2 = "sorted_centroid_dist"+str(x)+".txt"
    file2 = open(filename2,"wb")
    file2.write(str(sorted_centroid_dist))
    file2.close()

    #print "sorted_centroid_dist= ",sorted_centroid_dist


    #############################################################################################
    # Finding median and mean of words. 
    # Plotting the graph of distribution of words wtih their distance from centroid for analysis
    #############################################################################################

    # word = []
    # value = []
    # for i in range(len(sorted_centroid_dist)):
    #     #word.append((sorted_centroid_dist[i])[0])
    #     word.append(i)
    #     value.append((sorted_centroid_dist[i])[1])


    # j = len(sorted_centroid_dist)
    # median_value = value[j/2] 
    # print(median_value)
    # print(np.mean(value))

    # plt.scatter(word,value)
    # plt.ylabel('distance')
    # plt.xlabel('word')
    #plt.show()
    # plt.savefig("graph.png")


    ##############################################
    ############ FINDING TOP K ###################
    ##############################################
    

    k = 15
    top_k_terms = []
    for i in range(min(k,len(sorted_centroid_dist))):
        top_k_terms.append(str(sorted_centroid_dist[i][0]))
        
    print("Top k terms from feature selection are:")    
    print(top_k_terms)
    print("\n")


    ############################################
    ###########  Wikipedia #####################
    ############################################



    eightlist = []
    
    f = open("labels",'w')                          # This file will store all the candidate labels. 
    
    
    for i in range(len(top_k_terms)):
        term = top_k_terms[i]                       # picks top k terms found from centroid    
        keys = wikipedia.search(term)               # searches wikipedia to get titles of pages returned by wikipedia
        if not keys:                                # if nothing returned, leave.
            continue
        newlist = []                                # newlist will store new candidates
        try:
            newlist.append(str(keys[0]))            # It was found out that only first title returned was relevant enough.
            page = wikipedia.page(str(keys[0]))     # We query the whole page object from this first title
            templist = page.categories              # Find the categories from this page and add to templist
            f.write(str(keys[0])+"\n")              # Adding the title of first page to candidate labels file
            for j in range(len(templist)):          # adding all categories to the labels file as well.
                newlist.append(templist[j])
                f.write(templist[j]+"\n")
        
        except wikipedia.exceptions.DisambiguationError:     
            pass
    f.close()


 

    beautified_corpus=[]

    f = open('corpus', "r")

    # Removing irrelevant characters from the corpus file.
    for line in f :                                 
        line=line.strip()
        line=line.lower()
        line=re.sub("<.*?>-","",line)
        line = re.sub("[!,./;':\[\]\(\)\{\}\|<>?!#$%^&*]", " ", line)
        beautified_corpus.extend(line.split())

    f.close()

    beautified_corpus = [word.strip() for word in beautified_corpus]            # referred to shashank's code. Stripped the words again.


    f = open("labels",'r')          # contains all candidate labels.
    labels = f.readlines()          # adding all labels. 
    f.close()                       

    terms = top_k_terms[:]          # terms store all top_k_terms found using centroid method. Sort of a duplication. Can be removed.

    f.close()
    terms = [word.strip() for word in terms]        # beautifying top_k_terms
    labels = [word.strip() for word in labels]      # beautifying labels by wikipedia

    stoplist = []
    f=open('stopword.txt')
    for line in f :
        line=line[0:-1]             # from starting to the last.
        stoplist.append(line)       # making a stoplist of generic words which we don't want.
    f.close()
    

    lscore =[0 for x in range(len(labels))]         # lscore will keep a score for every candidate label.


    corp=[]                                                 # stores the whole corpus (beautified)
    corp = [word for word in a if word not in stoplist]     # beautifying corpus
    length = len(corp)                                      


    # writing vectors(inverse indexing) of top_k_terms in ranks.txt
    f = open("ranks.txt",'w')                                       
    for i in range(len(top_k_terms)):
        pickle.dump((newInverseIndexingDict[top_k_terms[i]]),f)
    f.close()

    # reading the file and making a ranks array which stores importance of top_k_terms. 
    f = open("ranks.txt",'r')
    ranks = []
    for i in range(len(top_k_terms)):
        test = pickle.load(f)
        sum = 0
        for j in range(len(test)):
            sum = sum + test[j]*test[j]
        ranks.append(math.sqrt(sum))

    #print(ranks)
    for j in range(len(ranks)):                 # Normalization to [0,1]
        ranks[j] = ranks[j]/length              # remove if results r fine without it.

    # Probability finding function!
    def pr(x,corpus):
        count=1
        for word in corpus:
          ## print word
            if x==word:
                count+=1
      ## print count
      #pdb.set_trace()
        l=len(corpus)
        return float (count)/l

    i=0
    from math import exp, expm1 , log
    #finding out the PMI score for each term in Candidate Label
    for label in labels:
        x=0
        pmi = 0;
        for term in terms:                 # So, it'll happen only top_k times. (terms contains 15 labels found from centroid)
            array = []
            array = [word for word in label.translate(None, '()[]-,?').lower().split() if word not in stoplist]     # extra-safe.
            array.append(term)

            num=0.0
            den=1.0
            for key in array:           # key is one of the top_k_terms
                num+=pr(key,corp)
                den*=pr(key,corp)
             

            temp=float(num)/den             # stores probability value
            temp*=float(ranks[x])           # multiplying importance of each top_k_terms with candidate labels.
            x+=1                            # for iteration. Used in ranks[x] in the above line.
            pmi += temp         
        lscore[i]= pmi                      # lscore stores the score of each candidate label from wikipedia
        i+=1

    # taking log of score and sorting it and storing in l_list (for all candidate labels)
    l_list = sorted([(log(x) if x > 0 else 0, y) for x, y in zip(lscore, labels)])
    # l_list_t stores only first 2
    l_list_t = l_list[:5]

    print("The label of the cluster is : {0}.{1}.{2}.{3.}.{4}".format(l_list_t[0],l_list_t[1],l_list_t[2],l_list_t[3],l_list_t[4]))

    with open('results1', 'w') as fp:       #results1 stores all the candidate labels with their pmi score
        for value, label in l_list_t:
            fp.write("%s %s\n" % (label, value))
  
    with open('results2', 'w') as fp:       #results2 stores only top 5 candidate labels with their pmi score
        for value, label in l_list:
            fp.write("%s %s\n" % (label, value))