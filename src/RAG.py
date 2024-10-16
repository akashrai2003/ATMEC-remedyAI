from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def get_query_from_disease(list_of_diseases: str,llmGetResponse):
    try:
        query = "List all the diseases and problems that the patient is suffering from"
        
        # print(context)
        prompt_template = """
        Question: {query}
        Context: {list_of_diseases}
        Based on the context provided that is a list of diseases, generate me a query which would actually tell more about the diet which tha patient should follow considering the diseases mentioned in the list.
        So the query should contain what kind of food should the patient eat and what kind of food should the patient avoid.

        Example: dietary recommendations for hypertension and diabetes.
        """

        prompt = prompt_template.format(list_of_diseases=list_of_diseases, query=query)

        prediction = llmGetResponse.invoke(prompt)
        print(prediction)
        prediction = prediction.content
        return prediction
    except Exception as e:
        print(f"Error occurred: {str(e)}")

def get_answer_from_query(query: str, qa_chain: RetrievalQA, retriever):
    # Retrieve context from the retriever
    context = retriever.get_relevant_documents(query)

    # If no relevant documents are found, return a default message
    if not context:
        return {"answer": "No relevant context found.", "source_document": "None", "doc": "Unknown"}

    # Prepare the input for the LLM using the expected key 'query'
    response = qa_chain({"query": query})

    # Extract the source document information if available
    if response['source_documents']:
        source_document = response['source_documents'][0].page_content
        doc = response['source_documents'][0].metadata.get('source', 'Unknown')
    else:
        source_document = "No source document found."
        doc = "Unknown"

    # Extract the answer
    answer = response['result']
    return {"answer": answer, "source_document": source_document, "doc": doc}
