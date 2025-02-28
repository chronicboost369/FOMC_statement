import requests
import pyperclip
import time 
from io import BytesIO
import PyPDF2
import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
import nltk 
import re
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


# %%

# Press Conference url
def generate_fomc_url(date_str):
   base_url = "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf"
   suffix = ".pdf"
   return f"{base_url}{date_str}{suffix}"


# Policy Statement
#def generate_fomc_url(date_str):
#   base_url = "https://www.federalreserve.gov/newsevents/pressreleases/monetary"
#   suffix = "a.htm"
#   return f"{base_url}{date_str}{suffix}"
#''''

# Historical FOMC dates
historical_fomc_dates= [
    20250129,
    20241218,
    20241107,
    20240918,
    20240731,
    20240612,
    20240501,
    20240320,
    20240131,
    20231213,
    20231101,
    20230920,
    20230726,
    20230614,
    20230503,
    20230322,
    20230201,
    20221214,
    20221102,
    20220921,
    20220727,
    20220615,
    20220504,
    20220316,
    20220126,
    20211215,
    20211103,
    20210922,
    20210728,
    20210616,
    20210428,
    20210317,
    20210127,
    20201216,
    20201105,
    20200916,
    20200729,
    20200610,
    20200429,
    20200331,
    20200319,
    20200315,
    20200303,
    20200129,
    20191211,
    20191030,
    20191004,
    20190918,
    20190731,
    20190619,
    20190501,
    20190320,
    20190130,
    20181108,
    20180926,
    20180801,
    20180613,
    20180502,
    20180321,
    20180131,
    20171213,
    20171101,
    20170920,
    20170726,
    20170614,
    20170503,
    20170315,
    20170201,
    20161214,
    20161102,
    20160921,
    20160727,
    20160615,
    20160427,
    20160316,
    20160127,
    20151216,
    20151028,
    20150917,
    20150729,
    20150617,
    20150429,
    20150318,
    20150128,
    20141217,
    20141029,
    20140917,
    20140730,
    20140618,
    20140430,
    20140319,
    20140304,
    20140129,
    20131218,
    20131030,
    20131016,
    20130918,
    20130731,
    20130619,
    20130501,
    20130320,
    20130130,
    20121212,
    20121024,
    20120913,
    20120801,
    20120620,
    20120425,
    20120313,
    20120125,
    20111213,
    20111102,
    20110921,
    20110809,
    20110622,
    20110427,
    20110315,
    20110126,
    20101214,
    20101103,
    20100921,
    20100810,
    20100623,
    20100428,
    20100316,
    20100127,
    20091216,
    20091104,
    20090923,
    20090812,
    20090624,
    20090429,
    20090318,
    20090128,
    20081216,
    20081029,
    20080916,
    20080805,
    20080625,
    20080430,
    20080318,
    20080130,
    20071211,
    20071031,
    20070918,
    20070807,
    20070628,
    20070509,
    20070321,
    20070131,
    20061212,
    20061025,
    20060920,
    20060808,
    20060629,
    20060510,
    20060328,
    20060131,
    20051213,
    20051101,
    20050920,
    20050809,
    20050630,
    20050503,
    20050322,
    20050202,
    20041214,
    20041110,
    20040921,
    20040810,
    20040630,
    20040504,
    20040316,
    20040128,
    20031209,
    20031028,
    20030916,
    20030812,
    20230625,
    20030506,
    20030318,
    20030129,
    20021210,
    20021106,
    20020924,
    20020813,
    20020626,
    20020507,
    20020319,
    20020130,
    20011211,
    20011106,
    20011002,
    20010821,
    20010627,
    20010515,
    20010320,
    20010131,
    20001219,
    20001115,
    20001003,
    20000822,
    20000628,
    20000516,
    20000321,
    20000202
]

urls = [generate_fomc_url(date) for date in historical_fomc_dates]



'''
for i in range(0,30):
    response = requests.get(urls[i])
    response.raise_for_status() 
    pdf_file = BytesIO(response.content)
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Now `text` contains the full text from the PDF
    print(text)
'''





records = []

# Loop over the dates to extract text from each PDF.
for date in range(0,(len(historical_fomc_dates))):
    url = generate_fomc_url(historical_fomc_dates[date])
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to retrieve {url}: {e}")
        continue  # Skip to the next date if there's an error.
    
    pdf_file = BytesIO(response.content)
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    

    records.append({
        "date": historical_fomc_dates[date],
        "url": url,
        "text": text
    })

'''
#loop for Policy statement
for date in historical_fomc_dates:  
    url = generate_fomc_url(date)
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to retrieve {url}: {e}")
        continue  # Skip to the next date if there's an error.
    
    # Parse the HTML content using BeautifulSoup.
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract the text. You might want to target specific sections depending on the page structure.
    text = soup.get_text(separator=' ', strip=True)
    
    # Append a record (dictionary) for each document.
    records.append({
        "date": date,
        "url": url,
        "text": text
    })
'''

# Create a DataFrame from the records.
df = pd.DataFrame(records)



# CSV for later use.
#df.to_csv("fomc_press_releases.csv", index=False)

# Getting Stock data
qqq = yf.download("QQQ", start="2001-01-01", end="2025-02-10",timeout=60)
#qqq.to_csv('qqq.csv',index=False)
qqq = pd.DataFrame(qqq).reset_index()
qqq = qqq.reset_index(drop=True)
qqq['daily_return'] = qqq['Close'].pct_change().fillna(0)
qqq['daily_vol'] = qqq['Close']/qqq['High']-1
qqq.columns = qqq.columns.get_level_values(0)

# Merge the stock data with the FOMC press release data.
# Adding a date(infinitely) to those fomc releases that came out during non-market days
# Looking for either the day of the release(or next day) and 3 market days after(a week)
days_after = 1
market_date = set(qqq['Date'])

def date_adjustment(date):
    while date not in market_date:
        date += pd.Timedelta(days=1)
    return date

df['date'] = pd.to_datetime(df['date'],format='%Y%m%d')
df['date'] = df['date'].apply(date_adjustment)
df['date_af'] = df['date']+pd.Timedelta(days=days_after)
df['date_af'] = df['date_af'].apply(date_adjustment) #adding dates to next available market open days if fed releases its doc on non-market open days
df['date'] = pd.to_numeric(df['date'].dt.strftime('%Y%m%d')) #converting time in yyyy-mm-dd to yyyymmdd
df['date_af'] = pd.to_numeric(df['date_af'].dt.strftime('%Y%m%d')) #converting time in yyyy-mm-dd to yyyymmdd
qqq['Date'] = pd.to_numeric(qqq['Date'].dt.strftime('%Y%m%d'))
qqq[f"{days_after}d_return"] = qqq['Close'].shift(-days_after)/qqq['Close']-1


df2 = pd.merge(df,qqq,left_on='date',right_on='Date') #merging corpus and qqq for the date of the release
df2 = df2.drop(['Date','Close','High','Low','Open','Volume','date_af'],axis=1)


#%%
# Building NLP Model

# Vectorization of the text
# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
# removing identified irrelevant common words to further clean the data using bag of words approach
remove_words = list(['chair', 'powell', 'press', 'conference', 'yellen','bernankes',
                'page','american','people','family','community',
                'january','feburary','march','april','may','june','july',
                'august','september','october','november','december','timiraos','nick','wall','journal'
                ,'street','29','26','powe','lls','chairman','chairwoman',
                'thank','look','forward','questions','would','say'])
fomc_years = [str(year) for year in np.unique([str(dates)[:4] for dates in historical_fomc_dates])]
remove_words.extend(fomc_years)

# tokenizing the text data
def text_tokenizer(x):
    x = x.lower() #converts all lower case
    x = re.sub(r'[^a-z0-9\s\u00BC-\u00BE\u2150-\u218F\.]', '', x) # removing punctuations and non_letters & numbers
    x2= re.sub(r'\.(?=\s)', '', x)

    tokens = word_tokenize(x2) #tokenizing using word_tokenize functions
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in remove_words ]
    return  " ".join(tokens)

df2['text'] = df2['text'].apply(text_tokenizer)


# %%
#EDA

# do bag of words to see the count
# then play around with max_df to filter out useless common words
vectorizer = TfidfVectorizer()
tfidf_text = vectorizer.fit_transform(df2['text'])
features = vectorizer.get_feature_names_out()
tfidf_dense = tfidf_text.toarray()
features[np.argmax(tfidf_dense.mean(axis=0))]


vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(df2['text'])
counts = bag_of_words.toarray()
counts = counts.sum(axis=0)
word = vectorizer.get_feature_names_out()
word_counts = pd.DataFrame({'word': word,'count':counts})

word_counts.sort_values(by='count',ascending=False)[0:20]
#Looking at top 20 words, it is certain that more words are needed to assess the Fed's tone.
# For example, 'inflation' alone has both positive and negative effects on the stock market.
# Inflation is essentially the fuel that brings the stock prices increase as the value of currency decreases.
# On the other hand, when inflation is elevated beyodn 2%, the Fed often increases the rate 
# to cool the economy. 

vectorizer = CountVectorizer(ngram_range=(2,2))
bag_of_words = vectorizer.fit_transform(df2['text'])
counts = bag_of_words.toarray()
counts = counts.sum(axis=0)
word = vectorizer.get_feature_names_out()
word_counts = pd.DataFrame({'word': word,'count':counts})
word_counts.sort_values(by='count',ascending=False)[0:20]
# still doesn't seem enough
# trying 4

vectorizer = CountVectorizer(ngram_range=(4,4))
bag_of_words = vectorizer.fit_transform(df2['text'])
counts = bag_of_words.toarray()
counts = counts.sum(axis=0)
word = vectorizer.get_feature_names_out()
word_counts = pd.DataFrame({'word': word,'count':counts})
word_counts.sort_values(by='count',ascending=False)[0:20]

vectorizer = CountVectorizer(ngram_range=(6,6))
bag_of_words = vectorizer.fit_transform(df2['text'])
counts = bag_of_words.toarray()
counts = counts.sum(axis=0)
word = vectorizer.get_feature_names_out()
word_counts = pd.DataFrame({'word': word,'count':counts})
word_counts.sort_values(by='count',ascending=False)[0:20]






# %%
# Modelling
# Train/Test Split -> Using data up to 20220630 as train and afterwards as test since this is a document that is released 7-8 times every year.

# Feature Extraction
# Starting out with TF-IDF as a baseline method of extraction for simplicity and capturing some key words that 
# may appear infrequently that may lose its importance in bag of words approach.
# This is an area of improvement for further improvement.

cutoff = 20220630
X_train = df2[df2['date']<=cutoff]['text']
X_test = df2[df2['date']>cutoff]['text']
Y_train = df2[df2['date']<=cutoff]['1d_return']
Y_test = df2[df2['date']>cutoff]['1d_return']

X_train.shape
Y_test.shape

# Pipeline
start = time.time()
time.sleep(2)
# Define a pipeline with a placeholder regressor.
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('regressor', LinearRegression())
])

# Define a parameter grid that includes different ngram ranges and four regressor types.
param_grid = [

    {
        'tfidf__ngram_range':  [(1, 3), (2, 5),(4,8)],
        'regressor': [Ridge()],
        'regressor__alpha': [0.1, 1.0, 10.0]
    },
    {
        'tfidf__ngram_range': [(1, 3), (2, 5),(4,8)],
        'regressor': [RandomForestRegressor(random_state=42)],
        'regressor__n_estimators': [50, 100],
        'regressor__max_depth': [10,20,40]
    },
    {
        'tfidf__ngram_range': [(1, 3), (2, 5),(4,8)],
        'regressor': [GradientBoostingRegressor(random_state=42)],
        'regressor__n_estimators': [50, 100,200],
        'regressor__learning_rate': [0.03,0.1 ,0.15],
        'regressor__max_depth': [10,20,40]
    }
]

# Use GridSearchCV to search over the parameter grid.
grid_search = GridSearchCV(pipeline,
                           param_grid,
                           cv=5,
                           scoring='neg_mean_squared_error',  # use negative MSE so that higher is better
                           verbose=1,
                           n_jobs=-1)

grid_search.fit(X_train, Y_train)
results = pd.DataFrame(grid_search.cv_results_) # results of cv for hyper tuning parameter

# best performing models are ridge and gbm with learning rate of 0.001, max depth = 20, tfidf_ngram_range = (1,3) n_estimator = 100, ngram=1,3

# Evaluating the best performing model on test data
test_result = []
for idx,row in results.iterrows():
    print(idx)
    param = row['params']
    model = pipeline.set_params(**param)
    model.fit(X_train,Y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_pred-Y_test)**2))
    test_result.append(rmse)


test_result_df = pd.concat([results['params'],pd.DataFrame(test_result,columns=['rmse'])],axis=1)
test_result_df.sort_values('rmse')

# Best Ridge
best_ridge_param = test_result_df[test_result_df['params'].astype(str).str.contains('Ridge')].sort_values('rmse',ascending=True).iloc[0,0]
best_ridge = pipeline.set_params(**best_ridge_param)
best_ridge.fit(X_train,Y_train)

best_ridge_coef = pd.DataFrame({
    'Var' : best_ridge.named_steps['tfidf'].get_feature_names_out(),
    'Coef' : best_ridge.named_steps['regressor'].coef_
})
best_ridge_coef.sort_values('Coef',ascending=False)

# Best RF
best_rf_param = test_result_df[test_result_df['params'].astype(str).str.contains('Random')].sort_values('rmse',ascending=True).iloc[0,0]
best_rf = pipeline.set_params(**best_rf_param)
best_rf.fit(X_train,Y_train)
best_rf_importance = pd.DataFrame({
    'Var' : best_rf.named_steps['tfidf'].get_feature_names_out(),
    'Importance' : best_rf.named_steps['regressor'].feature_importances_
})
best_rf_importance.sort_values('Importance',ascending=False).head(5)

# Best GBM
best_xgb_param = test_result_df[test_result_df['params'].astype(str).str.contains('Boosting')].sort_values('rmse',ascending=True).iloc[0,0]
best_xgb = pipeline.set_params(**best_xgb_param)
best_xgb.fit(X_train,Y_train)
best_xgb_importance = pd.DataFrame({
    'Var' : best_xgb.named_steps['tfidf'].get_feature_names_out(),
    'Importance' : best_xgb.named_steps['regressor'].feature_importances_
})
best_xgb_importance.sort_values('Importance',ascending=False).head(5)


# Combining all of the best test models of each algo.
combined_performance = pd.DataFrame(columns=['Model','Test_RMSE'])
combined_performance.loc[0,'Model'] = 'Ridge'
combined_performance.loc[1,'Model'] = 'RF'
combined_performance.loc[2,'Model'] = 'GBM'
combined_performance.loc[0,'Test_RMSE'] =  test_result_df[test_result_df['params'].astype(str).str.contains('Ridge')].sort_values('rmse',ascending=True).iloc[0,1]
combined_performance.loc[1,'Test_RMSE'] =  test_result_df[test_result_df['params'].astype(str).str.contains('Random')].sort_values('rmse',ascending=True).iloc[0,1]
combined_performance.loc[2,'Test_RMSE'] =  test_result_df[test_result_df['params'].astype(str).str.contains('Boosting')].sort_values('rmse',ascending=True).iloc[0,1]
plt.figure(figsize=(2, 1.5))
plt.bar(combined_performance['Model'], combined_performance['Test_RMSE'], color='blue')
plt.xlabel('Model')
plt.ylabel('Test RMSE')
plt.title('Test RMSE by Model')
plt.show()