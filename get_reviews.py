import pandas as pd

df = pd.read_csv('amazon-fine-food-reviews/Reviews.csv')
print(df.columns.values)
reviews = df['Text'].dropna().tolist()
print(reviews)
print(len(reviews))

with open('reviews.txt', 'w') as f:
    for review in reviews:
        f.write("%s\n" % review)
print("Done")
