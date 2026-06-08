#Question 1

#import libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation

# Load the dataset
df = pd.read_csv(r"C:\Users\reyna\Downloads\articles (1).csv", usecols=["id","year" ,"title", "abstract"])

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.shape)

#Data Cleaning
#-------------

#Remove duplicates
df = df.drop_duplicates()

#Replace "Abstract Missing" with an empty string
df["abstract_clean"] = df["abstract"].replace("Abstract Missing", "", regex=False)

#Fill missing text values
df["title"] = df["title"].fillna("")
df["abstract_clean"] = df["abstract_clean"].fillna("")

#Combine title and abstract into a single text column
df["text_source"] = df["title"] + " " + df["abstract_clean"]

#ensure year is numeric
df["year"] = pd.to_numeric(df["year"], errors="coerce")

# Drop rows with missing or invalid year
df = df.dropna(subset=["year"])
df = df[df["text_source"].str.strip() != ""]

df["year"] = df["year"].astype(int)

print("Rows after cleaning:", df.shape[0])
print("Years range from", df["year"].min(), "to", df["year"].max())
print("Abstracts missing count:", (df["abstract"] == "Abstract Missing").sum())


#Exploratory Data Analysis
#-------------------------
print(df.columns)
papers_per_year = df.groupby("year").size().reset_index(name="count")

plt.figure(figsize=(10, 6))
plt.plot(papers_per_year["year"], papers_per_year["count"], marker='o')
plt.title("Number of Papers Published Each Year")
plt.xlabel("Year")
plt.ylabel("Number of Papers")
plt.grid(True)
plt.tight_layout()
plt.show()

#Text Length Analysis
df["text_length"] = df["text_source"].apply(lambda x: len(x.split()))

plt.figure(figsize=(10, 6))
plt.hist(df["text_length"], bins=30, edgecolor='black')
plt.title("Distribution of Text Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print(df["text_length"].describe())


#Text Preprocessing
#------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text_clean"] = df["text_source"].apply(clean_text)

#Custom stop words
custom_stopwords = {
    "study", "results", "conclusion", "method", "methods", "data", "analysis",
    "research", "paper", "findings", "significant", "model", "models",
    "approach", "approaches", "propose", "proposed", "use", "used",
    "using", "based", "new", "novel", "machine", "learning", "algorithm", "algorithms",
    "data"
}

stop_words = list(ENGLISH_STOP_WORDS.union(custom_stopwords))

#Converting text to a document-term matrix
vectorizer = CountVectorizer(
    max_df=0.95,
    min_df=2,
    max_features=3000,
    stop_words=stop_words,
    ngram_range=(1, 2)
)

dtm = vectorizer.fit_transform(df["text_clean"])
print("Document-Term Matrix shape:", dtm.shape)


#Determine optimal number of topics using perplexity
#----------------------------------------------------

topic_options = [6, 8, 10, 12]

tuning_results = []

for k in topic_options:
    lda = LatentDirichletAllocation(
        n_components=k,
        random_state=42,
        learning_method="online",
        learning_decay=0.7,
        max_iter=8,
        batch_size=256
    )
    
    lda.fit(dtm)
    
    log_likelihood = lda.score(dtm)
    perplexity = lda.perplexity(dtm)
    
    tuning_results.append({
        "topics": k,
        "log_likelihood": log_likelihood,
        "perplexity": perplexity
    })

tuning_df = pd.DataFrame(tuning_results)
print(tuning_df)


#Final LDA Model training
#-------------------------

optimal_topics = 6

final_lda = LatentDirichletAllocation(
    n_components=optimal_topics,
    random_state=42,
    learning_method="online",
    learning_decay=0.7,
    max_iter=8,
    batch_size=256
)

lda_output = final_lda.fit_transform(dtm)

# Assign dominant topic to each paper
df["dominant_topic"] = lda_output.argmax(axis=1)

print(df[["year", "title", "dominant_topic"]].head())


#Display top words for each topic
#---------------------------------

feature_names = vectorizer.get_feature_names_out()

def display_topics(model, feature_names, number_of_words=15):
    topics = {}
    
    for topic_index, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-number_of_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        topics[f"Topic {topic_index}"] = top_words
        
        print(f"\nTopic {topic_index}:")
        print(", ".join(top_words))
    
    return topics

topic_keywords = display_topics(final_lda, feature_names)

#Visualize number of papers per topic
topic_counts = df["dominant_topic"].value_counts().sort_index()

plt.figure(figsize=(8, 6))
plt.bar(topic_counts.index.tolist(), topic_counts.to_list())
plt.title("Number of Papers per Dominant LDA Topic")
plt.xlabel("Topic")
plt.ylabel("Number of Papers")
plt.tight_layout()
plt.show()

#Visualize topic distribution over years
topic_trends = df.groupby(["year", "dominant_topic"]).size().reset_index(name="count")

year_totals = df.groupby("year").size().reset_index(name="total")

topic_trends = topic_trends.merge(year_totals, on="year")

topic_trends["percentage"] = (
    topic_trends["count"] / topic_trends["total"]
) * 100

topic_pivot = topic_trends.pivot(
    index="year",
    columns="dominant_topic",
    values="percentage"
).fillna(0)

plt.figure(figsize=(12, 7))

for topic in topic_pivot.columns:
    plt.plot(topic_pivot.index, topic_pivot[topic], label=f"Topic {topic}")

plt.title("Topic Popularity Over Time")
plt.xlabel("Year")
plt.ylabel("Percentage of Papers")
plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

#topic keywords table
topic_table = []

for topic_name, words in topic_keywords.items():
    topic_table.append({
        "Topic": topic_name,
        "Top Words": ", ".join(words)
    })
    
topic_df = pd.DataFrame(topic_table)
print(topic_df)