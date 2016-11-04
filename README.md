# Cluster-Labelling-using-wikipedia
K-means clustering was done using scikit-learn library. This is followed by extraction of important keywords using tf-idf centroid. Top 'k' labels from here are sent to wikipedia as search query. The search results are used to generate better labels.
All candidate labels were judged using Mutual Information judge and a high accuracy of 85.6% was obtained for 20-Newsgroups dataset. Dataset is not included as a part of the repo.

Note : The code is still all in one file, but is documented pretty well through comments.


