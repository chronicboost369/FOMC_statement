# FOMC_statement

## Intro

The phrase "Don't fight the Fed" is well known in the stock market, reflecting the immense influence of the Federal Reserve, the United States' central bank. By managing interest rates and its balance sheet, the Fed controls the US economy and stock market towards desirable direction. A striking example unfolded in March 2020, when the COVID-19 pandemic triggered a market crash. The Fed’s swift response—slashing interest rates and expanding its balance sheet—paired with federal stimulus packages, fueled a remarkable "V"-shaped recovery, propelling the U.S. stock market into a historic bull run. That momentum persisted until 2022, when the Fed’s shift to a tighter monetary policy sparked a sharp downturn.

This project explores whether data from FOMC press releases can be harnessed to model and predict short-term stock market movements, leveraging data science to decode the Fed’s outsized impact.

## Data

For this data science project, I scraped data directly from the Federal Reserve website, covering the period from 2000 to 2025. Although the Fed typically conducts eight scheduled meetings annually, variations in URL formats across different batches of press releases limited our scrape to 81 documents. The collected data spans from April 2011 to January 2025, forming the foundation for our analysis.

To preprocess the text data, I first remove stop words and apply the bag-of-words method to identify the most frequently used terms across the FOMC press releases. During this step, words like "forward," "thank," and "would" are filtered out—though common, they carry little meaningful insight for our analysis. Next, we use TF-IDF (Term Frequency-Inverse Document Frequency) to convert the text into numerical values. This approach is ideal because it downweights terms that appear frequently across all press releases, emphasizing words that are more distinctive and informative for specific documents.

For the response variable, I calculate the one-day performance of QQQ, the NASDAQ 100 ETF, defined as the ratio of the closing price one day after the FOMC press release to the closing price on the release date itself. I chose QQQ for its simplicity and near-perfect correlation with the NASDAQ 100 index. The NASDAQ 100 is preferred over the S&P 500 or Dow Jones because its constituent stocks are more responsive to shifts in economic conditions. Focusing on one-day performance helps isolate the market’s reaction to the FOMC statement, minimizing interference from unrelated news events that might coincidentally occur around the same time.

## Modelling

The algorithms under consideration are: 1. ridge regression 2. random forest 3.gradient boosting machine (GBM). 

Ridge regression was selected from a range of parametric regression techniques because it effectively manages multicollinearity through shrinkage parameters. This is particularly useful since TF-IDF can generate a vast number of variables, and ridge regression helps by increasing bias to reduce variance, stabilizing the model. On the other hand, ensemble tree-based methods—random forest and GBM—were chosen for their ability to capture nonlinear relationships, which is critical given the limited exploratory data analysis (EDA) possible with text data to uncover links between explanatory variables (TF-IDF features) and the response variable (QQQ performance). Random forest excels at controlling variance, while GBM focuses on minimizing bias, offering complementary strengths. Comparing these two ensemble methods alongside ridge regression should provide valuable insights into the underlying relationships and enhance the project’s development.

The overall structure of modelling is preprocessing the text data using TF-IDF then hyperparameter tunings are conducted using CV(cross validation) approach (SKlearn). Test RMSE is selected to compare the performance of models. Test RMSE is chosen as it has the same unit as the response variables thus making interpretability straightforward. Additionally, due to limited computing power, only a small subset of hyperparameters are tested for tuning purposes.


## Result

### Ridge Regression

Hypertuned Parameters and tested values:
1. tfidf__ngram_range:  [(1, 3), (2, 5),(4,8)]
2. L2 penalty: [0.1, 1.0, 10.0]

Selected model:
1. tfidf__ngram_range: (1,3)
2. L2 penalty: 10

Test RMSE:
  0.4929%

Key Variables
In the ridge regression model, I analyzed terms based on the magnitude of their coefficients, distinguishing between positive and negative values. The five terms with the largest positive coefficients do not appear to reveal significant nuances in the Federal Reserve's statements that could influence market behavior. In contrast, the two of five terms with the largest negative coefficients, such as "inflation" and "rate," are closely tied to the Fed’s policy language. However, these terms alone do not fully clarify the Fed’s implications, as their market impact—whether bullish or bearish—hinges on the context provided by surrounding words in the statement. As a result, it is difficult to confirm that the ridge regression model has effectively pinpointed the key terms driving short-term market movements.

![image](https://github.com/user-attachments/assets/40029b72-0e48-4628-b2d5-aab154df9f39)

### Random Forest

Hypertuned Parameters and tested values:
1. tfidf__ngram_range: [(1, 3), (2, 5),(4,8)]
2. n_estimators: [50, 100]
3. max_depth: [10,20,40]

Selected model:
1. tfidf__ngram_range: (2,5)
2. n_estimators: 100
3. max_depth: 10

Test RMSE: 0.4882%

Key Variables
Key variables are chosen based on their importance, a metric that measures how much each variable reduces the model's errog. Surprisingly, the top five most important variables in the model are all tied to the Federal Reserve’s policies. These variables reflect the Fed’s positions on unemployment rates and interest rate hikes. However, there’s a limitation to this approach: the importance measure in random forests doesn’t reveal the direction of the relationship between these variables and the response variable. Even with this limitation, these findings could be valuable as unemployment and rate hikes are major macroeconomic indicators. Keeping an eye on these factors could help predicting short-term market movements. Additionally, unlike Ridge regression, all 5 of these key variables are related to the economy.

![image](https://github.com/user-attachments/assets/dc076d38-51a1-4686-8bc6-6853f5519681)


### GBM

Hypertuned Parameters and tested values:
1. tfidf__ngram_range: [(1, 3), (2, 5),(4,8)]
2. n_estimators': [50, 100,200]
3. learning_rate': [0.03,0.1 ,0.15]
4. max_depth': [10,20,40]

Selected model:
1. tfidf__ngram_range: (2,5)
2. n_estimators: 50
3. learning_rate: 0.03
4. max_depth: 10

Test RMSE: 0.603%

  



