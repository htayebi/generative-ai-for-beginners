from openai import AzureOpenAI
import os
import dotenv

# import dotenv
dotenv.load_dotenv()

# configure Azure OpenAI service client 
client = AzureOpenAI(
  azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"], 
  api_key=os.environ['AZURE_OPENAI_API_KEY'],  
  api_version = "2023-10-01-preview"
  )

deployment=os.environ['AZURE_OPENAI_DEPLOYMENT']

# add your completion code
prompt = "List Prime Ministers from 2010 to 2020" 

response = client.chat.completions.create(
    model=deployment,
    messages=[
        {
            "role": "system",
            "content": "You are an Australian historian.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ],
    # Optional parameters
    temperature=1.,
    max_tokens=1000,
    top_p=1.    
)

# print response
print(response.choices[0].message.content)