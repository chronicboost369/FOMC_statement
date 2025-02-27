# FOMC_statement

## Intro

The phrase "Don't fight the Fed" is well known in the stock market, reflecting the immense influence of the Federal Reserve, the United States' central bank. By managing interest rates and its balance sheet, the Fed controls the US economy and stock market towards desirable direction. A striking example unfolded in March 2020, when the COVID-19 pandemic triggered a market crash. The Fed’s swift response—slashing interest rates and expanding its balance sheet—paired with federal stimulus packages, fueled a remarkable "V"-shaped recovery, propelling the U.S. stock market into a historic bull run. That momentum persisted until 2022, when the Fed’s shift to a tighter monetary policy sparked a sharp downturn.

This project explores whether data from FOMC press releases can be harnessed to model and predict short-term stock market movements, leveraging data science to decode the Fed’s outsized impact.

## Data

For this data science project, we scraped data directly from the Federal Reserve website, covering the period from 2000 to 2025. Although the Fed typically conducts eight scheduled meetings annually, variations in URL formats across different batches of press releases limited our scrape to 81 documents. The collected data spans from April 2011 to January 2025, forming the foundation for our analysis.

To preprocess the text data, after removing stop words, bag of words method is first utilized to evaulate most commonly used words. During this process words like "forward", "thank", and "would" are removed as they are commonly used but doesn't really contain any value. Then, TF-IDF (Term Frequency - Inverse Document Frequency) method is used to transform text data to numeric values.
Such method is chosen to remove commonly used words across press releases as it penealizes the more commonly used terms.

For the response variable, QQQ, NASDAQ 100 ETF, 1 day performance (FOMC press release date + 1 day close price / press releaste date close price) is utilized. QQQ is chosen for simplicity since it is almost perfectly corrleated to NASDAQ 100 index. MASDAQ 100 is chosen over S&P500 or DOW Jones because it is consist of stocks that are more sensitive to economic conditions. Finally, 1 day performance is evaluated to separate from impactful news that could be concidentally released around FOMC meetings. 
