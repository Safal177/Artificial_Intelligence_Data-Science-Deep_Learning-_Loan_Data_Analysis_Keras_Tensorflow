# Artificial_Intelligence_Data-Science-Deep_Learning-_Loan_Data_Analysis_Keras_Tensorflow

# Using Deep Learning (Lending Club Data) for Loan Default Prediction

# Objective: 
With historical loan data from 2007 to 2015, the goal of this project was to make a projective model which can pinpoint whether a loan offered by “Lending Club” will credit risk or not. This project carried out implementing exploratory data analysis to identify borrower and loan attributes, encoding categorical features into numerical form, and implementing feature engineering to trim duplication and correlation among features. A deep learning model was then developed using Keras with Tensorflow trained on the preprocessed dataset, and examined with various metrics such as confusion matrix, and classification reports, and roc-auc. 

# Project file: Notebook
Artificial_Intelligence_Data Science_Deep_Learning_Loan_Data_Analysis_Keras_Tensorflow.jpynb

# Technologies used:
•	Python
•	NumPy, Pandas, Matplotlib, Seaborn
•	Scikit-learn
•	Keras and TensorFlow

# Conclusion: 
The study demonstrated that the loan dataset is significantly uneven, with notably a smaller number of loan defaults relative to completely paid loans. After preprocessing and variable filtering, the deep learning model accomplished a “roc-auc” score of ~ 0.66, highlighting intermediate forecasting proficiency. Even though the model revealed significant detection rate (~0.76) for loan defaults, it underperformed with correctness (~0.21) which indicates the difficulties of class asymmetry in financial datasets. This project shows the deep learning for credit uncertainty projections and focusing further necessary refinements through innovative methods such as resampling and boosting approaches to improve predictive accuracy.
