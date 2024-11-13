from openai import AzureOpenAI
import os
import dotenv
import numpy as np

# import dotenv
dotenv.load_dotenv()

# configure Azure OpenAI service client 
client = AzureOpenAI(
  azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"], 
  api_key=os.environ['AZURE_OPENAI_API_KEY'],  
  api_version = "2023-10-01-preview"
  )

deployment=os.environ['AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT']


automobile_embedding = client.embeddings.create(input='automobile', model=deployment).data[0].embedding
vehicle_embedding = client.embeddings.create(input='vehicle', model=deployment).data[0].embedding
fox_embedding = client.embeddings.create(input='the quick brown fox jumped over the lazy dog', model=deployment).data[0].embedding
poo_embedding = client.embeddings.create(input='poo', model=deployment).data[0].embedding

result = np.dot(poo_embedding, vehicle_embedding) / (np.linalg.norm(poo_embedding) * np.linalg.norm(vehicle_embedding))

print(result)
