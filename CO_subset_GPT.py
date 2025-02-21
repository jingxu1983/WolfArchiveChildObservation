#!/usr/bin/env python
# coding: utf-8

# In[1]: 
## This notebook free for educational reuse under Creative Commons CC BY License.
## Created by Grant Glass for the 2024 Text Analysis Pedagogy Institute, with support from Constellate.


### Install Libraries ###
get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install openai pandas matplotlib tiktoken')


# In[3]:


### Import Libraries ###
# Import the openai library for accessing OpenAI's API functionalities
import openai
# Import the OpenAI class from the openai library for more specific API interactions
from openai import OpenAI
# Import the os library to interact with the operating system, like reading or writing files
import os
# Import pandas, a powerful data manipulation and analysis library, as 'pd'
import pandas as pd
# Import matplotlib's pyplot to create static, interactive, and animated visualizations in Python, as 'plt'
import matplotlib.pyplot as plt
# Import the constellate library for working with datasets and analytics
# Import time
import time
# Import tiktoken which helps us count tokens
import tiktoken


# In[11]:


print(os.getcwd())


# In[12]:


### Import my dataset
df = pd.read_csv("data/SBERT_Sample.csv")
print(df.head())


# In[14]:


###Configure the OpenAI client

#To setup the client for our use, we need to create an API key to use with our request. 
## Method 1: Directly paste the API key (not recommended for production or shared code)
client = OpenAI(api_key= "enter your own API key")


# In[15]:


# Define a function `get_completion` that takes a prompt and optionally a model name (defaulting to "gpt-3.5-turbo")
def get_completion(prompt, model="gpt-3.5-turbo"):
    # Create a list of messages where each message is a dictionary with a role (user/system) and the content (the prompt)
    messages = [{"role": "user", "content": prompt}]
    # Call the OpenAI API's chat.completions.create method with the specified model, messages, and a temperature of 0
    # Temperature of 0 makes the model's responses deterministic (the same input will always produce the same output), max 2.
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    # Return the content of the first message in the response's choices. This is the model's completion of the input prompt.
    return response.choices[0].message.content

# Test the function by defining a test prompt asking to explain what a large language model is in one sentence
test_prompt = "Explain children's play in two sentence."
# Print the result of calling `get_completion` with the test prompt to see the model's response
print(get_completion(test_prompt))


# In[17]:


### LLM performance
# Define a function to calculate the number of tokens in a string using a specific encoding
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    # Retrieve the encoding object for the specified encoding name using tiktoken library
    encoding = tiktoken.get_encoding(encoding_name)
    # Encode the string and calculate the number of tokens in the encoded result
    num_tokens = len(encoding.encode(string))
    # Return the number of tokens
    return num_tokens

# Define a function to analyze a text in parts, each part having a maximum number of tokens
def analyze_text_in_parts(text, task, max_tokens=8000):
    # Check if the input text is a list (of strings) or a single string
    if isinstance(text, list):
        # If it's a list, join the elements into a single string separated by spaces
        text = ' '.join(text)
    
    # Initialize an empty list to hold the parts of text
    parts = []
    # Initialize an empty string to accumulate the current part of text
    current_part = ""
    # Split the text into words
    words = text.split()
    
    # Iterate over each word in the text
    for word in words:
        # Check if adding the current word to the current part exceeds the max_tokens limit
        if num_tokens_from_string(current_part + " " + word) > max_tokens:
            # If so, add the current part to the parts list and start a new part with the current word
            parts.append(current_part)
            current_part = word
        else:
            # Otherwise, add the current word to the current part
            current_part += " " + word
    
    # After iterating through all words, add the last part to the parts list if it's not empty
    if current_part:
        parts.append(current_part)
    
    # Initialize an empty string to accumulate the aggregated response from analyzing each part
    aggregated_response = ""
    # Iterate over each part
    for part in parts:
        # Construct a prompt for the analysis task using the current part
        prompt = f"Analyze the following text for {task}: {part}"
        # Get the response for the prompt using the get_completion function
        response = get_completion(prompt)
        # Append the response to the aggregated response, separated by spaces
        aggregated_response += response + " "
        # Sleep for 1 second to avoid hitting rate limits of the API
        time.sleep(1)
    
    # Return the aggregated response, stripped of leading/trailing whitespace
    return aggregated_response.strip()

# Define a list of tasks for analysis (can replace the following tasks with "sentiment analysis", "topic models", etc.)
tasks = ["sentiment", "topic models"]
# Initialize an empty list to store the results
results = []

# Iterate over each row in the dataframe `df`
for _, row in df.iterrows():
    # For each task, analyze the full text of the row
    for task in tasks:
        analysis = analyze_text_in_parts(row['Content'], task)
        # Append the analysis result to the results list
        results.append({"Text": row['ObsIndex'], "Task": task, "Analysis": analysis})
    
    # After analyzing all tasks for a row, save the results to a CSV file to prevent data loss
    pd.DataFrame(results).to_csv('ChildPlay1_analysis.csv', index=False)

# Convert the results list to a DataFrame and print it
df_results = pd.DataFrame(results)
print(df_results)


# In[22]:


### LLM performance
# Define a function to calculate the number of tokens in a string using a specific encoding
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    # Retrieve the encoding object for the specified encoding name using tiktoken library
    encoding = tiktoken.get_encoding(encoding_name)
    # Encode the string and calculate the number of tokens in the encoded result
    num_tokens = len(encoding.encode(string))
    # Return the number of tokens
    return num_tokens

# Define a function to analyze a text in parts, each part having a maximum number of tokens
def analyze_text_in_parts(text, task, max_tokens=8000):
    # Check if the input text is a list (of strings) or a single string
    if isinstance(text, list):
        # If it's a list, join the elements into a single string separated by spaces
        text = ' '.join(text)
    
    # Initialize an empty list to hold the parts of text
    parts = []
    # Initialize an empty string to accumulate the current part of text
    current_part = ""
    # Split the text into words
    words = text.split()
    
    # Iterate over each word in the text
    for word in words:
        # Check if adding the current word to the current part exceeds the max_tokens limit
        if num_tokens_from_string(current_part + " " + word) > max_tokens:
            # If so, add the current part to the parts list and start a new part with the current word
            parts.append(current_part)
            current_part = word
        else:
            # Otherwise, add the current word to the current part
            current_part += " " + word
    
    # After iterating through all words, add the last part to the parts list if it's not empty
    if current_part:
        parts.append(current_part)
    
    # Initialize an empty string to accumulate the aggregated response from analyzing each part
    aggregated_response = ""
    # Iterate over each part
    for part in parts:
        # Construct a prompt for the analysis task using the current part
        prompt = f"Analyze the following text for {task}: {part}"
        # Get the response for the prompt using the get_completion function
        response = get_completion(prompt)
        # Append the response to the aggregated response, separated by spaces
        aggregated_response += response + " "
        # Sleep for 1 second to avoid hitting rate limits of the API
        time.sleep(1)
    
    # Return the aggregated response, stripped of leading/trailing whitespace
    return aggregated_response.strip()

# Define a list of tasks for analysis (can replace the following tasks with "sentiment analysis", "topic models", etc.)
tasks_new = ["classification as cooperation in score 0-10", "classification as conflict in score 0-10", "classification as pretend-play in score 0-10"]
# Initialize an empty list to store the results
results_new = []

# Iterate over each row in the dataframe `df`
for _, row in df.iterrows():
    # For each task, analyze the full text of the row
    for task_new in tasks_new:
        analysis = analyze_text_in_parts(row['Content'], task_new)
        # Append the analysis result to the results list
        results_new.append({"Text": row['ObsIndex'], "Task": task_new, "Analysis": analysis})
    
    # After analyzing all tasks for a row, save the results to a CSV file to prevent data loss
    pd.DataFrame(results_new).to_csv('ChildPlay1_analysis_new.csv', index=False)

# Convert the results list to a DataFrame and print it
df_results_new = pd.DataFrame(results_new)
print(df_results_new)


# In[24]:


# concatenate strings
def concat_strings(values):
    return ', '.join(values)


# In[33]:


#make long tables wide
df1 = df_results.pivot_table(index='Text', columns='Task', values='Analysis', aggfunc=concat_strings)
print(df1.head())
pd.DataFrame(df1).to_csv('ChildPlay1_df1.csv', index=True)


# In[34]:


#make long tables wide
df2 = df_results_new.pivot_table(index='Text', columns='Task', values='Analysis', aggfunc=concat_strings)
print(df2.head())
pd.DataFrame(df2).to_csv('ChildPlay1_df2.csv', index=True)


# In[35]:


# Extract a column from the first DataFrame
merged_df = df2.merge(df1, on='Text', how = 'outer')
print(merged_df.head())
pd.DataFrame(merged_df).to_csv('ChildPlay1.csv', index=True)


# In[36]:


pd.DataFrame(merged_df).to_excel('ChildPlay1.xlsx', index=True)






