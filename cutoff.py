import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Load the dataset
file_path = 'processed_dataset_50k.csv'
df = pd.read_csv(file_path)

# Calculate the length of each sentence in both English and Portuguese
df['english_length'] = df['English'].str.split().str.len()
df['portuguese_length'] = df['Portuguese'].str.split().str.len()

# Analyze the distribution for English
print("English Sentence Lengths:")
print(df['english_length'].describe())

# Analyze the distribution for Portuguese
print("\nPortuguese Sentence Lengths:")
print(df['portuguese_length'].describe())

# Plotting the distribution for English
sns.histplot(df['english_length'], bins=30, kde=True)
plt.xlabel('English Sentence Length')
plt.ylabel('Frequency')
plt.title('Distribution of English Sentence Lengths')
plt.show()

# Plotting the distribution for Portuguese
sns.histplot(df['portuguese_length'], bins=30, kde=True)
plt.xlabel('Portuguese Sentence Length')
plt.ylabel('Frequency')
plt.title('Distribution of Portuguese Sentence Lengths')
plt.show()

# Determine the 95% cutoff for English
english_cutoff = df['english_length'].quantile(0.95)
english_cutoff_rounded = math.ceil(english_cutoff)
print(f"Rounded 95% cutoff for English sentence length: {english_cutoff_rounded} words")

# Determine the 95% cutoff for Portuguese
portuguese_cutoff = df['portuguese_length'].quantile(0.95)
portuguese_cutoff_rounded = math.ceil(portuguese_cutoff)
print(f"Rounded 95% cutoff for Portuguese sentence length: {portuguese_cutoff_rounded} words")
