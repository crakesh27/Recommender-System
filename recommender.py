import pandas as pd
import numpy as np
import math
import pickle


def load_pkl(s):
    """
    Loads the pickle

    """
    pkl_file = open(s, 'rb')
    data = pickle.load(pkl_file)
    return data


def write_pkl(multimap_list, s):
    """
    Pickles the given file

    """
    writing = open(s, 'wb')
    pickle.dump(multimap_list, writing)
    return

class CollaborativeFiltering():
    """
        Predicts the ratings of first quater of user movie matrix and
        calculates RMSE, Rank Correlation using Collaborative Filtering
    """
    def __init__(self, rating_matrix):
        self.rating_matrix = rating_matrix
        self.num_users = self.rating_matrix.shape[0]
        self.num_movies = self.rating_matrix.shape[1]

        self.mean_corrected_matrix = self.user_rating_mean_correction()
        self.similarity_matrix = self.get_user_user_similarity_matrix()

    def user_rating_mean_correction(self):

        mean_corrected_matrix = self.rating_matrix.copy()

        for i in range(self.num_users):
            try:
                user_mean = self.rating_matrix[i].sum(
                )/len(np.flatnonzero(self.rating_matrix[i]))
            except:
                user_mean = 0

            mean_corrected_matrix[i] = np.where(
                self.rating_matrix[i] != 0, self.rating_matrix[i] - user_mean, 0)
        return mean_corrected_matrix

    def user_similarity(self, i, j):

        num = np.dot(
            self.mean_corrected_matrix[i], self.mean_corrected_matrix[j])
        m1 = math.sqrt(
            np.dot(self.mean_corrected_matrix[i], self.mean_corrected_matrix[i]))
        m2 = math.sqrt(
            np.dot(self.mean_corrected_matrix[j], self.mean_corrected_matrix[j]))

        try:
            return num/(m1*m2)
        except:
            return 0

    def get_user_user_similarity_matrix(self):

        try:
            similarity_matrix = load_pkl('user_sum.pkl')
            return similarity_matrix
        except:
            similarity_matrix = np.ndarray((self.num_users, self.num_users))
            for i in range(self.num_users):
                for j in range(self.num_users):
                    similarity_matrix[i][j] = self.user_similarity(i, j)

            write_pkl(similarity_matrix, 'user_sum.pkl')
            return similarity_matrix

    def predict_rating(self, x, i):
        num = 0.0
        denom = 0.0
        for y in range(self.num_users):
            if x == y:
                continue

            if self.similarity_matrix[x][y] > 0 and self.rating_matrix[y][i] != 0:
                num += self.similarity_matrix[x][y] * self.rating_matrix[y][i]
                denom += self.similarity_matrix[x][y]

        if denom != 0:
            return num/denom
        else:
            try:
                return self.rating_matrix[x].sum()/len(np.flatnonzero(self.rating_matrix[x]))
            except:
                return 0

    def RMSEandSpearman(self):
        diff = 0.0
        num_pred = 0
        rows = self.num_users // 4
        cols = self.num_movies // 4
        for i in range(rows):
            for j in range(cols):
                if self.rating_matrix[i][j] != 0:
                    diff += ((self.predict_rating(i, j) -
                             self.rating_matrix[i][j])**2)
                    num_pred += 1

        rmse = math.sqrt(diff/num_pred)
        rankcor = 1-((6*diff)/(num_pred*((num_pred**2)-1)))

        print('Collaborative RMSE', rmse)
        print('Collaborative Spearmans Rank Correlation', rankcor)

    def precisionOnTopK(self):
        """
            Precision On Top K for Collaborative filtering
        """
        k = 50
        relevance = 3

        sum = 0.0
        tot = 0

        ind = np.argsort(-1 * self.rating_matrix, axis = 1)

        for i in range(self.num_users):

            num, den = 0.0, 0.0
            for j in range(k):
                movie_index  = ind[i][j]
                if(self.rating_matrix[i][movie_index] >= relevance):
                    den += 1
                    pred_value = self.predict_rating(i, movie_index)
                    if(pred_value >= relevance):
                        num += 1
            
            try:
                val = num/den
            except:
                val = 0
        
            sum += val
            tot += 1
        
        print('Collaborative Precision On TopK', sum/tot)


def get_user_movie_rating_matrix():
    """
    Loads data from u1.base file and constructs the user_movie_rating_matrix

    """
    df = pd.read_csv("u1.base", sep="\t")
    num_row = df['userId'].max()
    num_col = df['movieId'].max()

    user_movie_rating_matrix = np.ndarray(shape=(num_row, num_col))

    for i in range(len(df)):
        user_movie_rating_matrix[int(
            df['userId'][i])-1][int(df['movieId'][i])-1] = float(df['rating'][i])

    return user_movie_rating_matrix


if __name__ == "__main__":

    user_movie_rating_matrix = get_user_movie_rating_matrix()

    recommender = CollaborativeFiltering(user_movie_rating_matrix)
    recommender.RMSEandSpearman()
    recommender.precisionOnTopK()
    """
        Collaborative RMSE 0.8452675603293487
        Collaborative Spearmans Rank Correlation 0.9999999240526046
        Collaborative Precision On TopK 0.9091970463737955
    """
