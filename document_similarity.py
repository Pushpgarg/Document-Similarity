from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np
import os
os.system('clear')

load_dotenv()

# both ways hugging face and en
# embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "who was former indian caption"

document_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

similarity_score = cosine_similarity([query_embedding] , document_embedding)[0]      # because it will return a 2d list 
   
print(documents[np.argmax(similarity_score)])
