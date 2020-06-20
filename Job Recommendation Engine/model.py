import pandas as pd
import numpy as np
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# reading the data from the preprocessed .csv file
data = pd.read_csv('indeed_data.csv')

data['Output'] = data[['Title', 'Company', 'Location','Salary']].apply(lambda x: '/ '.join(x), axis=1)
# making the new column containing combination of all the features
data['soup']=data['Description']+data['Title']


# creating a count matrix
cv = CountVectorizer(analyzer='word',ngram_range=(1,3), stop_words='english')
count_matrix = cv.fit_transform(data['soup'])

# creating a similarity score matrix
sim = cosine_similarity(count_matrix,count_matrix)

# saving the similarity score matrix in a file for later use
np.save('similarity_matrix', sim)

# saving dataframe to csv for later use in main file
data.to_csv('indeed_data.csv',index=False)