# Library to get token count for a sentence
import tiktoken
import pandas as pd
import matplotlib.pyplot as plt
import os
import openai
from dotenv import load_dotenv

load_dotenv()
# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")
openai.api_key = os.environ.get("OPENAI_API_KEY")
max_tokens = 500

def get_df():
    df = pd.read_csv('processed/scraped.csv', index_col=0)
    df.columns = ['title', 'text']
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    return df

# df = pandas dataframe type, assumes we have a column `n_tokens`
def visualize_token_count_histogram(df):
    # Visualize the distribution of the number of tokens per row using a histogram
    plt.hist(df['n_tokens'], bins=20)  # Adjust the number of bins as needed
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.title('Histogram of n_tokens')
    plt.show()

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks

def raw_df_to_shortened(df):
    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'])
        
        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append( row[1]['text'] )
    
    return shortened

def create_shortened_df(shortened):
    df = pd.DataFrame(shortened, columns = ['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    return df

def create_embeddings_column(df):
    df['embeddings'] = df.text.apply(
        lambda x: openai.Embedding.create(
            input=x, 
            engine='text-embedding-ada-002'
        )['data'][0]['embedding'])
    df.to_csv('processed/embeddings.csv')

###### EXECUTION BLOCK ######

raw_df = get_df()
shortened = raw_df_to_shortened(raw_df)
shortened_df = create_shortened_df(shortened)
create_embeddings_column(shortened_df)