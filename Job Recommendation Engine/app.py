import numpy as np
import pandas as pd
from flask import Flask, render_template, request
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_sim():
    data = pd.read_csv('indeed_data.csv')
    # creating a count matrix
    cv = CountVectorizer(analyzer='word',ngram_range=(1,3), stop_words='english')
    count_matrix = cv.fit_transform(data['soup'])
    # creating a similarity score matrix
    sim = cosine_similarity(count_matrix,count_matrix)
    return data,sim

def rcmd(m):
    #m = m.lower()
    # check if data and sim are already assigned
    try:
        data.head()
        sim.shape
    except:
        data, sim = create_sim()
    # check if the movie is in our database or not
    if m not in data['Job Title'].unique():
        return('This Job is not in our database.\nPlease check if you spelled it correct.')
    else:
        # getting the index of the movie in the dataframe
        idx = data.loc[data['Job Title']==m].index[0]
       

        # fetching the row containing similarity scores of the movie
        # from similarity matrix and enumerate it
        lst = list(enumerate(sim[int(idx)]))

        # sorting this list in decreasing order based on the similarity score
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)

        # taking top 1- movie scores
        # not taking the first index since it is the same movie
        lst = lst[1:11]

        # making an empty list that will containg all 10 movie recommendations
        
        
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['Output'][a])
        return l


    
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')



@app.route("/recommend")
def recommend():
    job = request.args.get('job')
    r = rcmd(job)
    job = job.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',job=job,r=r,t='s')
    else:
        return render_template('recommend.html',job=job,r=r,t='l')

if __name__ == '__main__':
    app.run()




