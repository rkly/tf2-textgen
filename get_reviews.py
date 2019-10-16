import pandas as pd

# read
df = pd.read_csv('amazon-fine-food-reviews/Reviews.csv')
print(df.columns.values)

# clean
df = df[df.Score > 2]
df_reviews = df['Text'].dropna().drop_duplicates()
reviews = df_reviews.tolist()
#print(reviews)
print(len(reviews))

with open('reviews.txt', 'w') as f:
    for review in reviews:
        f.write("%s\n" % review)
print("Done")