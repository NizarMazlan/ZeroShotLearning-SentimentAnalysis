
import streamlit as st
import pandas as pd
import time
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
from streamlit_option_menu import option_menu
# Start
#st.title('Zero-Shot Learning in Sentiment Analysis')
#st.text('test')

st.sidebar.title('Navigation')

with st.sidebar:
    option = option_menu(
        menu_title = 'Main Menu',
        options = ['Home','Zero-Shot Learning','Food Review Dataset','Financial News Dataset','Movie Review Dataset','Game Review Dataset','😀 Sentiment Analyser 🤬'],
        icons = ['house','app','egg-fried','cash','film','controller'],
        menu_icon = 'cast',
        default_index = 0,
    )

#option = st.sidebar.selectbox('Pages', options = ['👋 Home','0️⃣ Zero-Shot Learning 0️⃣','🍔 Food Review Dataset 🍔','💵 Financial News Dataset 💵','🎥 Movie Review Dataset 🎥','🎮 Game Review Dataset 🎮','😀 Sentiment Analyser 🤬'])

if option == 'Home':
    # Main Description
    st.markdown("# 👋 Welcome to Zero-Shot Learning in Sentiment Analysis")
    st.markdown("Developed by __Mohamad Nizar Mustaqeem__")
    st.markdown("The app is still under development. Please reach me in the github repo if you have any comments or suggestions.")

    # Description of the Panels
    st.markdown(
    """
    ### Select on the left panel what you want to explore:
    - With the Zero-Shot Learning panel, you will be shown a a simple zero-shot learning project specifically in sentiment analysis.
    - With the datasets panel you will be exploring the dataset. It includes some analysis of the data and models trained using the datasets : 
        - 🍔 Food Review Dataset 🍔
        - 💵 Financial News Dataset 💵 
        - 🎥 Movie Review Dataset 🎥
        - 🎮 Game Review Dataset  🎮
    - With 😀 Sentiment Analyser 🤬, you will explore the created models by using Bag of Words Vectorization-Based Models.
    \n  
    
    """
)

if option == 'Zero-Shot Learning':
    st.title('0️⃣ Zero-Shot Learning in Sentiment Analysis')
    st.header('Introduction')
    st.markdown("""Hello and welcome to this page of my (Mohamad Nizar Mustaqeem) data science project. The title of my project is Zero-Shot Learning in Sentiment Analysis. Basically zero-shot Learning is a setup in which a model can learn to recognize things that it hasn't explicitly seen before in training. In real life situation, new concepts, topics and fields emerge, a trained model would be obsolete as unforeseen and new data will always be created""")
    st.header('Objective')
    st.write("1. To investigate the feasibility of zero-shot learning in sentiment analysis")
    st.write("2. To evaluate the sentiment analysis based on zero-shot learning")
    st.header('Dataset')
    st.image("Images/Datasets.png")
    st.write("To demonstrate the concept of Zero-Shot Learning in Sentiment Analysis, 4 different types of dataset is being used. Amazon Fine Food Reviews, Financial Phrasebank, IMDB Movie Reviews and last but not least Steam Game Reviews. These datasets includes reviews and sentences with their sentiments")
    st.header('Methodology')
    st.image("Images/Methodology.png")
    st.markdown(""" For Data Collection, the main sources of the datasets are from Kaggle. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers.

    Next for Data Cleaning and Preprocessing, removal of null values and duplicates is being done. Transformation of data to standardize accorss all datasets. Then whitespaces is unified, punctuations are removed and the data is being cleaned of emojis, basic characters and numbers. Last but not least is the removal of stopwords and the act of stemming.

    Then proceed with Building Models. Three different algorithm is being used in this project, Support Vector Machine (SVM), Logistic Regression (LR), and Random Forest (RF). Each model will be trained and tested on each of the datasets as much as five times. 

    For Evaluation, each model's result created will be collected and mean of it will be used to evaluate which dataset perform well and did not. Confusion matrix is used to calculate the accuracy. The result will be tabulated for a better Visualization""")
    st.header('Exploratory Data Analysis')
    st.markdown("""
    ### Distribution
    """)
    st.image("Images/EDA_Distribution.png")
    st.write("As for the distribution, we can see some of the dataset have an extremely unbalanced distribution. It might affect the accuracy of the model")
    st.markdown("""
    ### Word Cloud
    """)
    st.image("Images/EDA_WordCloud.png")
    st.write("From this word cloud, there are still words that might be interfere with out model building. It needs to be cleaned. But from this word cloud we can see that the frequent words are catered to each dataset domain.")
    st.header('Results and Findings')
    st.image("Images/Results1.png")
    st.write("From the test results of each models evaluation, it is tabulated using Tableau. The highlight table on the left shown indicated that generally model trained and tested from the same dataset have a high accuracy if not the highest. On the line chart on the right, we can see that the three algorithm used basically have a similar performance accuracy.")
    st.image("Images/Results2.png")
    st.write("Next, the test results are being calculated as a mean and categorized as Non Zero-Shot results and Zero-Shot. Zero-Shot is a condition where test data is not seen or available during model building. Here we can see that Non Zero-Shot test results has the highest accuracy. But one dataset stands out which is Food Review. It has a high accuracy when tested on unseen data and we can see it more clearly on the line chart.")
    st.header('Conclusion')
    st.markdown(
    """
    - Zero-shot learning is a setup in which a model can learn to recognize things that it hasn't explicitly seen before in training.
    - From the built models and it's test accuracy is lower when tested on an unseen dataset. Although Food Review Dataset performed well when trained and tested on other dataset.
    - Faced a challenge of having an unbalanced data in all of the dataset, result tend to be biased and some datesets does not include a neutral sentiment.
    - Future works can consider having a more comprehensive parameter-tuning and using other approaches
    """
)
if option == 'Food Review Dataset':
    st.title('🍔 Food Review')
    st.header('Introductory')
    st.write('This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories. [Link to Dataset](%s)' % "https://www.kaggle.com/code/laowingkin/amazon-fine-food-review-sentiment-analysis/data")
    st.header('Exploratory Data Analysis')
    AFF_Revs = pd.read_csv(r'C:\Users\User\University Malaya\Sem 5\Data Science Project\WebApp\Data\AmazonReview.csv')
    st.write(AFF_Revs.head())
    st.write('Distribution Of The Score')
    st.image("Images/foodRevScore.png")
    st.write('WordCloud')
    st.image("Images/foodRevWordCloud.png")
    st.header('Sentiment Analysis')
    st.image("Images/FOODTESTEXDATA.png")


if option == 'Financial News Dataset':
    st.title('💵 Financial News Dataset')
    st.header('Introductory')
    st.write('This dataset (FinancialPhraseBank) contains the sentiments for financial news headlines from the perspective of a retail investor.The dataset contains two columns, "Sentiment" and "News Headline". The sentiment can be negative, neutral or positive. [Link to Dataset](%s)' % "https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news?datasetId=622510&sortBy=voteCount")
    st.header('Exploratory Data Analysis')
    Fin_PB = pd.read_csv(r'C:\Users\User\University Malaya\Sem 5\Data Science Project\WebApp\Data\FinancialPhraseBank.csv',encoding="ISO-8859-1")
    st.write(Fin_PB.head())
    st.write('Distribution Of The Sentiment')
    st.image("Images/FinPBSentiment.png")
    st.write('WordCloud')
    st.image("Images/FinPBWordCloud.png")
    st.header('Sentiment Analysis')
    st.image("Images/FINEXDATA.png")


if option == 'Movie Review Dataset':
    st.title('🎥 Movie Review Dataset')
    st.header('Introductory')
    st.write('IMDB dataset having 50K movie reviews for natural language processing or Text analytics. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. So, predict the number of positive and negative reviews using either classification or deep learning algorithms. [Link to Dataset](%s)' % "https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format?select=Train.csv")
    st.header('Exploratory Data Analysis')
    IMBD_Revs = pd.read_csv(r'C:\Users\User\University Malaya\Sem 5\Data Science Project\WebApp\Data\MovieReview.csv')
    st.write(IMBD_Revs.head())
    st.write('Distribution Of The Label')
    st.image("Images/MovieRevLabel.png")
    st.write('WordCloud')
    st.image("Images/MovieRevWordCloud.png")
    st.header('Sentiment Analysis')
    st.image("Images/MOVIEEXDATA.png")

if option == 'Game Review Dataset':
    st.title('🎮 Game Review Dataset')
    st.header('Introductory')
    st.write('Steam is a video game digital distribution service with a vast community of gamers globally. A lot of gamers write reviews on the game page and have the option of choosing whether they would recommend this game to others or not. However, determining this sentiment automatically from the text can help Steam to automatically tag such reviews extracted from other forums across the internet and can help them better judge the popularity of games. Given the review text with user recommendation and other information related to each game for 64 game titles, the task is to create a test set by making a split from the training set and try to predict whether the reviewer recommended the game titles available in the test set on the basis of review text and other information.')
    st.header('Exploratory Data Analysis')
    Game_Revs = pd.read_csv(r'C:\Users\User\University Malaya\Sem 5\Data Science Project\WebApp\Data\GameReview.csv')
    st.write(Game_Revs.head())
    st.write('Distribution Of The Label')
    st.image("Images/GameRevUserSugg.png")
    st.write('WordCloud')
    st.image("Images/GameRevWordCloud.png")
    st.header('Sentiment Analysis')
    st.image("Images/GAMEEXDATA.png")

if option == '😀 Sentiment Analyser 🤬':
    st.title('😀 Sentiment Analyser 🤬')
    st.write('This is a custom built sentiment analyser created using 3 algorithms trained using 4 different datasets')
    with st.form(key = 'nlpForm'):
        raw_text = st.text_area("Enter your sentence here")
        submit_button = st.form_submit_button(label = 'Analyze')
        col1,col2 = st.columns(2)
        # load the model from disk
        pkl_FoodRev = pickle.load(open('Saved Model Pickle/FoodRev_LR.pkl', 'rb'))
        pkl_Fin = pickle.load(open('Saved Model Pickle/Fin_LR.pkl', 'rb')) 
        pkl_MovieRev = pickle.load(open('Saved Model Pickle/Movie_LR.pkl', 'rb'))  
        pkl_GameRev = pickle.load(open('Saved Model Pickle/Game_LR.pkl', 'rb'))

        pkl_FoodRevSVM = pickle.load(open('Saved Model Pickle/FoodRev_SVM.pkl', 'rb'))
        pkl_FinSVM = pickle.load(open('Saved Model Pickle/Fin_SVM.pkl', 'rb'))
        pkl_MovieRevSVM = pickle.load(open('Saved Model Pickle/Movie_SVM.pkl', 'rb'))
        pkl_GameRevSVM = pickle.load(open('Saved Model Pickle/Game_SVM.pkl', 'rb'))

        pkl_FoodRevRF = pickle.load(open('Saved Model Pickle/FoodRev_RF.pkl', 'rb'))
        pkl_FinRF = pickle.load(open('Saved Model Pickle/Fin_RF.pkl', 'rb')) 
        pkl_MovieRevRF = pickle.load(open('Saved Model Pickle/Movie_RF.pkl', 'rb'))
        pkl_GameRevRF = pickle.load(open('Saved Model Pickle/Game_RF.pkl', 'rb'))
    
        if submit_button:

            with st.spinner('Wait for it...'):
                time.sleep(3)
            st.success('Done!')

            with col1:
                st.info("Results")
                resultFoodLR = pkl_FoodRev.predict([raw_text])
                resultFinLR = pkl_Fin.predict([raw_text])
                resultMovieRevLR = pkl_MovieRev.predict([raw_text])
                resultGameRevLR = pkl_GameRev.predict([raw_text])

                resultFoodSVM = pkl_FoodRevSVM.predict([raw_text])
                resultFinSVM = pkl_FinSVM.predict([raw_text])
                resultMovieRevSVM = pkl_MovieRevSVM.predict([raw_text])
                resultGameRevSVM = pkl_GameRevSVM.predict([raw_text])

                resultFoodRF = pkl_FoodRevRF.predict([raw_text])
                resultFinRF = pkl_FinRF.predict([raw_text])
                resultMovieRevRF = pkl_MovieRevRF.predict([raw_text])
                resultGameRevRF = pkl_GameRevRF.predict([raw_text])

                data = [[resultFoodLR,resultFinLR,resultMovieRevLR,resultMovieRevLR],
                [resultFoodSVM,resultFinSVM,resultMovieRevSVM,resultGameRevSVM],
                [resultFoodRF,resultFinRF,resultMovieRevRF,resultGameRevRF]]
                df = pd.DataFrame(data,columns=['Food Review Model','Financial Model', 'Movie Review Model', 'Game Review Model'])
                df.index = ['Logistic Regression', 'SVM', 'Random Forest']
                st.table(df)
                st.markdown(
                    """
                    1 : Positive Sentiment\n
                    0 : Negative Sentiment
                    """)
                


                # Emoji for the results
                #if result == 1:
                    #st.markdown("The sentence is Positive :smiley: ")
                #elif result == 0:
                   # st.markdown("The sentence is Negative :angry: ")