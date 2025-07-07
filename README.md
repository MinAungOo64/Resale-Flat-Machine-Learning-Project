# Resale-Flat-Machine-Learning-Project
Machine learning project to predict future flat resale prices using SciKit Learn library  

This project aims to predict future flat resale prices.  
The original datasets can be found here https://data.gov.sg/collections/189/view

---
## EDA and Data Processing
I will highlight some important steps below regarding data processing and explain the rationale behind each decision.

### Selecting the data
Data between 2012 to 2025 are selected for training. Resale prices before 2012 are based on when the transactions are aprroved while resale prices after 2012 are based on when transactions are registered. Furthermore, we do not want the models to learn outdated price trends.

### Resale Price over the years

![image](https://github.com/user-attachments/assets/fbdb89bd-11e6-4ab4-b7c4-f8f996505071)


```python
url1 = "https://raw.githubusercontent.com/MinAungOo64/Resale-Flat-Machine-Learning-Project/refs/heads/main/Resale%20Flat%20Prices%20(Based%20on%20Registration%20Date)%2C%20From%20Mar%202012%20to%20Dec%202014.csv"
url2 = "https://raw.githubusercontent.com/MinAungOo64/Resale-Flat-Machine-Learning-Project/refs/heads/main/Resale%20Flat%20Prices%20(Based%20on%20Registration%20Date)%2C%20From%20Jan%202015%20to%20Dec%202016.csv"
url3 = "https://raw.githubusercontent.com/MinAungOo64/Resale-Flat-Machine-Learning-Project/refs/heads/main/Resale%20flat%20prices%20based%20on%20registration%20date%20from%20Jan-2017%20onwards.csv"
```


---

The data is first processed in "EDA and Data Processing.ipynb" and saved as .pkl files.  
Two dataframes were created, "df_base" contains data from 2012 to 2025 and "df_post_covid" contains data post covid-19.  
The processed data is then fed into multiple machine learning models in each .ipynb files.  
For further details, click on each ipynb files.
