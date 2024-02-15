# import csv

# # Path to the input CSV file
# input_csv_path = 'dataset.csv'
# # Path to the output CSV file
# output_csv_path = 'cleaned_dataset.csv'

# # Headers for the output CSV file
# headers = ["English", "Portuguese"]

# # List to store the first 5 cleaned lines for printing
# first_five_cleaned_lines = []

# # Read the dataset from the CSV file, clean it, and write to a new CSV file
# with open(input_csv_path, 'r', encoding='utf-8') as infile, \
#         open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile:

#     # Create a CSV reader and writer
#     reader = csv.reader(infile)
#     writer = csv.writer(outfile)

#     # Write the header to the output CSV
#     writer.writerow(headers)

#     # Go through each row in the input CSV, count for first 5 lines
#     line_count = 0

#     for row in reader:
#         # Split the row on the tab character, which seems to be the delimiter in your data
#         parts = row[0].split('\t') if row else []

#         # Clean the English phrase
#         english_phrase = parts[0].split('CC-')[0].strip() if parts else ''

#         # Clean the Portuguese phrase if it exists
#         portuguese_phrase = parts[1].split('CC-')[0].strip() if len(parts) > 1 else ''

#         # Assemble the cleaned row
#         cleaned_row = [english_phrase, portuguese_phrase]

#         # Write the cleaned row to the output CSV
#         writer.writerow(cleaned_row)

#         # Keep the first five cleaned lines
#         if line_count < 5:
#             first_five_cleaned_lines.append(cleaned_row)
#             line_count += 1

# # Print the first 5 cleaned lines
# print("First 5 cleaned lines:")
# for line in first_five_cleaned_lines:
#     print(', '.join(line))

# print(f'Cleaned data saved to {output_csv_path}')


# import pandas as pd

# # Path to your TSV file
# file_path = 'tatoeba_dataset.tsv'

# # Read the TSV file
# # Assuming the columns are: ID, English, ID, Portuguese
# # Read the CSV file
# df = pd.read_csv(file_path, sep='\t', header=None, names=['ID1', 'English', 'ID2', 'Portuguese'], error_bad_lines=False, warn_bad_lines=True)

# # Drop the ID columns
# df.drop(['ID1', 'ID2'], axis=1, inplace=True)

# # Limit to 50,000 rows
# df = df.head(50000)

# # Save as CSV
# df.to_csv('tatoeba_dataset_50k.csv', index=False)

# print("File saved as 'tatoeba_dataset_50k.csv'")



# import pandas as pd

# def cut_dataset_in_half(file_path):
#     # Read the CSV file into a DataFrame
#     data = pd.read_csv(file_path)

#     # Calculate the midpoint of the dataset
#     midpoint = len(data) // 2

#     # Split the dataset into two halves
#     first_half = data.iloc[:midpoint]
#     second_half = data.iloc[midpoint:]

#     # Save each half into new CSV files
#     first_half.to_csv('first_half.csv', index=False)
#     second_half.to_csv('second_half.csv', index=False)

#     print("Dataset has been split into two files: 'first_half.csv' and 'second_half.csv'")

# # Path to your dataset file
# file_path = 'processed_dataset_50k.csv'

# # Call the function with your file path
# cut_dataset_in_half(file_path)



# import pandas as pd

# def reduce_dataset_size(file_path, reduction_factor=10):
#     # Carregar o arquivo CSV em um DataFrame
#     data = pd.read_csv(file_path)

#     # Selecionar aleatoriamente uma fração dos dados
#     reduced_data = data.sample(frac=1/reduction_factor)

#     # Salvar os dados reduzidos em um novo arquivo CSV
#     reduced_data.to_csv('reduced_dataset.csv', index=False)

#     print(f"O dataset foi reduzido e salvo em 'reduced_dataset.csv'")

# # Caminho para o arquivo CSV
# file_path = 'first_half.csv'

# # Chamar a função com o caminho do arquivo
# reduce_dataset_size(file_path)


# import pandas as pd

# # Load the dataset
# # Replace 'path_to_your_dataset.csv' with the actual file path of your dataset
# dataset = pd.read_csv('first_half.csv')

# # Cut off to 10k rows
# dataset_10k = dataset.head(10000)

# # Save to a new CSV file
# # You can change 'reduced_dataset.csv' to your preferred file name
# dataset_10k.to_csv('10k_dataset.csv', index=False)



import pandas as pd

# Load the dataset
df = pd.read_csv('tokenized_dataset_Medium.csv')

# Drop the columns 'english_tokens' and 'spanish_tokens'
df = df.drop(['english_tokens', 'spanish_tokens'], axis=1)

# Save the modified dataset to a new CSV file
df.to_csv('modified_dataset.csv', index=False)
