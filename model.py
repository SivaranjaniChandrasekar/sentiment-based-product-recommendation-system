from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd

pickle_folder = 'models/'
dataset_folder = 'data/'
class SentimentBasedProductRecommendationSystem:
    def __init__(self):
        self.data = self.read_pickle(dataset_folder + 'df_clean_data.pkl')
        self.user_final_rating = self.read_pickle(pickle_folder + 'User_based_recommendation_model.pkl')
        self.logistic_regression_model =  self.read_pickle(pickle_folder + 'Logistic_regression_Model.pkl')


    def read_pickle(self, file_path):
        return pickle.load(open(file_path,'rb'))

    def recommendProducts(self, user_name):
        items = self.user_final_rating.loc[user_name].sort_values(ascending=False)[0:20]
        features = self.read_pickle(pickle_folder + 'tfidf_model.pkl')
        top_20_df = pd.DataFrame(items)
        top_20_df.reset_index(inplace = True)
        top_20_sentiment = pd.merge(top_20_df, self.data, on = ['name'])
        vectorizer = TfidfVectorizer(vocabulary = features)
        top20_tfidf = vectorizer.transform(top_20_sentiment["reviews_text"])
        top20_pred = self.logistic_regression_model.predict(top20_tfidf)
        top_20_sentiment['pred_sentiment_score'] = top20_pred
        sentiment_score = top_20_sentiment.groupby(['name'])['pred_sentiment_score'].agg(['sum', 'count']).reset_index()
        sentiment_score['percentage'] = round((100*sentiment_score['sum'] / sentiment_score['count']),2)
        sentiment_score = sentiment_score.sort_values(by = 'percentage',ascending = False)
        op = sentiment_score['name'].head(5)
        return self.data.to_html(index=False)