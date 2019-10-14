import pandas as pd

df = pd.read_csv('amazon-fine-food-reviews/Reviews.csv')
print(df.columns.values)
reviews = df['Text'].dropna().tolist()
print(reviews)
print(len(reviews))