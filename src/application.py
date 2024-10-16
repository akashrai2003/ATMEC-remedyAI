import os
import traceback
import json
import io
import aiofiles
import logging
import numpy as np
from pydantic import BaseModel
from typing import Dict, List, Optional
from fastapi import FastAPI, File, Form, UploadFile, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from data_loader import load_documents
from upload import upload_data
from diagnosis import get_diagnosis_CXR, get_medLLM_response, process_image_and_query, process_image_query_and_json, get_medLLM_response_general
from diagnosis import get_diagnosis_dermat, compare_responses, web_query, web_query_RAG, web_empowered_results, web_empowered_results_RAG, compare_responses_dermat, compare_responses_general
from tavily_search import web_search
from GANsImage import generate_images_gan_chest, generate_images_gan_knee
from GANsTabular import breast_GAN, diabetes_GAN, heart_GAN
from RAG import get_answer_from_query, get_query_from_disease

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_TYPE = os.getenv("AZURE_OPENAI_API_TYPE")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # Update the environment variable name
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HF_INFERENCE_API_KEY = os.getenv("HF_INFERENCE_API_KEY")
GEOLOCATION_API= os.getenv("GEOLOCATION_API")


# Set up embeddings for GPU usage
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize Chroma vector store
persist_directory = r"A:\Downloads\Remedy.ai\store\doc_cosine"
vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)


# Initialize FastAPI
app = FastAPI()

# Configure CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Ensure all variables are correctly set
if not all([OPENAI_API_KEY, OPENAI_API_TYPE, AZURE_ENDPOINT, OPENAI_API_VERSION]):
    raise ValueError("Some environment variables are missing.")

# Instantiate the AzureChatOpenAI model using LangChain with the correct parameters
llmGetResponse = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,  # Use azure_endpoint instead of openai_api_base
    openai_api_key=OPENAI_API_KEY,
    openai_api_version=OPENAI_API_VERSION,
    openai_api_type=OPENAI_API_TYPE,
    deployment_name="Test",  # Ensure this is your correct deployment name
    temperature=0.7
)
cwd = os.getcwd()
print(cwd)
IMAGEDIR = 'A:/Downloads/Remedy.ai/images'
os.makedirs(IMAGEDIR, exist_ok=True)
# Initialize Pinecone
pc=Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(name=PINECONE_INDEX_NAME)

logging.info(index.describe_index_stats())

# Function to embed text using HuggingFace API
def embed_text(texts):
    API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_INFERENCE_API_KEY, model_name="BAAI/bge-large-en-v1.5", api_url=API_URL
    )
    if isinstance(texts, str):
        texts = [texts]
    embedding = embeddings.embed_documents(texts)
    return embedding

logging.info("Vector store successfully initialized with Pinecone and LangChain.")

namespace = 'default'

class Query(BaseModel):
    query: str

class Namespace(BaseModel):
    set_namespace: str

class User(BaseModel):
    lat: float
    long: float    

class DoctorRequest(BaseModel):
    specialization: str
    location: str
    user: User

@app.post("/set_namespace")
async def set_namespace(setNamespace: Namespace):
    global namespace
    namespace = setNamespace.set_namespace
    logging.info(f"Namespace set to {namespace}")
    return {"message": "Namespace set successfully", "namespace": namespace}

@app.post("/upload/files")
async def upload_files(
    files: List[UploadFile],
    metadata: str = Form(...),
):
    total_docs = []
    save_dir = "datasets"
    os.makedirs(save_dir, exist_ok=True)

    for file in files:
        file_path = os.path.join(save_dir, file.filename)
        content = await file.read()
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
        await file.close()

        if not isinstance(metadata, dict):
            metadata = json.loads(metadata)

        docs = load_documents(
            file_name=file.filename,
            file_path=file_path,
            chunk_size=1000,
            metadata_obj=metadata,
        )
        total_docs.extend(docs)
    try:
        upload_data(total_docs, namespace)
        return {"message": "Files uploaded successfully", "metadata": metadata}, 200
    except Exception as e:
        traceback.print_exc()
        logging.error(f"Error uploading files,\n {traceback.format_exc()}")
        raise e

@app.post("/get_disease")
async def get_disease():
    try:
        query = "List all the diseases and problems that the patient is suffering from"
        logging.info(f"Received query: {query}")
        query_values = embed_text(query)
        query_response = index.query(
            top_k=50,
            vector=query_values,
            include_values=True,
            include_metadata=True,
            namespace=namespace,
        )

        context = ""
        for match in query_response["matches"]:
            context += match["metadata"]["text"] + "\n"

        max_tokens = 3000
        context_tokens = context.split()
        if len(context_tokens) > max_tokens:
            context = ' '.join(context_tokens[:max_tokens])

        logging.info(f"Constructed context: {context}")
        # print(context)
        prompt_template = """
        Context: {context}
        Question: {question}

        Based on the context provided, list all the diseases and problems that the patient is suffering from. 
        Do not include diseases or problems that are explicitly stated as negative or absent. 
        Things to return:
        - If empty, then return an empty list [].
        - If not empty, then return the list of diseases and problems in square brackets like this -> [disease1, disease2, disease3] and don't give any other thing inside or outside the answer except the disease names which is mentioned that the patient is suffering from it, else you will get fired from your role.
        
        """
        prompt = prompt_template.format(context=context, question=query)

        prediction = llmGetResponse.invoke(prompt)
        print(prediction)
        prediction = prediction.content
        logging.info(f"Generated answer: {prediction}")

        response_data = {"answer": prediction}
        return JSONResponse(response_data)
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return JSONResponse(content={"error": str(e)})

@app.post("/get_response")
async def get_response(user_query: Query):
    try:
        query = user_query.query
        logging.info(f"Received query: {query} and index name as {index}")
        query_values = embed_text(query)
        query_response = index.query(
            top_k=10,
            vector=query_values,
            include_values=True,
            include_metadata=True,
            namespace=namespace,
        )

        context = ""
        for match in query_response["matches"]:
            context += match["metadata"]["text"] + "\n"

        max_tokens = 3800
        context_tokens = context.split()
        if len(context_tokens) > max_tokens:
            context = ' '.join(context_tokens[:max_tokens])

        logging.info(f"Constructed context: {context}")
        
        prompt_template = """
        You are a highly knowledgeable and specialized medical professional with expertise in interpreting and analyzing medical reports. Your task is to provide precise, detailed, and helpful answers based on the provided medical context. Ensure your responses are strictly confined to the medical information contained in the context and avoid any general or unrelated information. Your answers should be detailed, relevant to the user's query, and easy for amateurs to understand, explaining medical terms and concepts where necessary.

        Context: {context}

        Question: {question}

        Guidelines:
        1. If the context retrieved has no relevant information to the medical/healthcare domain, then answer the query from your own medical knowledge for user's benefits.
        2. Carefully consider the medical context provided and ensure your answer is directly derived from this context if relevancy is found in it with the user's query.
        3. If the context mentions multiple conditions but clarifies some as negative, it means clearly that the patient isn't suffering from that disease but it was being tested and the patient doesn't have that disease.
        4. Provide detailed medical advice or information relevant to the question, ensuring it aligns with the medical data in the context.
        5. Explain medical terms and concepts in a way that is easy for amateurs to understand.
        6. Provide detailed answers that are comprehensive and educational when asked by query.
        7. Do not provide answers outside the medical domain or unrelated to the context.
        8. If asked a question regarding any other domain like general knowledge, technical knowledge in software domains, etc., then answer that you are specialized in the medical domain and can't provide answers outside the medical domain.
        9. If you don't know the answer or if the question is from any other domains excluding medical/healthcare, just say that you are not specialized in that domain and thus don't know the answers, don't try to make up an answer.

        Answer:
        """
        prompt = prompt_template.format(context=context, question=query)

        prediction = llmGetResponse.invoke(prompt)
        prediction = prediction.content
        logging.info(f"Generated answer: {prediction}")

        response_data = {"answer": prediction}
        return JSONResponse(response_data)
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return JSONResponse(content={"error": str(e)})


#Recommendations

# Define the prompt template
prompt_template = """
Based on the disease list , I want to make create a query which would define what kind of food should a person intake based on his history of disease.
Then I want to get the list of foods which could be included in the diet based on the nutritional values from the vectorDB suitable for for the query .

For example, if:
query: Dietary plan for people with diabetes and high blood pressure -->
output: the food which could be included in the diet based on the nutritional values from the vectorDB suitable for a diet of a patient with diabetes and high BP.

Context: {context}
Question: {question}

Please list the relevant foods based on the above Question:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# Set up the retriever and QA chain
retriever = vector_store.as_retriever(search_kwargs={"k": 1})
chain_type_kwargs = {"prompt": prompt}
qa_chain = RetrievalQA.from_chain_type(
    llm=llmGetResponse,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    verbose=True
)

# Pydantic model for request body
class QueryRequest(BaseModel):
    query: str

# FastAPI endpoint for querying
@app.post("/get_answer/")
async def get_answer_based():

    try:
        # Call the get_disease endpoint to get the list of diseases
        disease_response = await get_disease()

        # Extract the answer from the disease response
        disease_list = disease_response.body.decode('utf-8')  # decode JSON response
        # Assuming the response format is {"answer": "[disease1, disease2]"}
        # You can parse this as needed
        logging.info(f"Using disease list as input: {disease_list}")
        query= get_query_from_disease(disease_list, llmGetResponse)
        response = get_answer_from_query(query, qa_chain, retriever)
        seach_query = web_query_RAG(response)
        web_results = web_search(seach_query)
        output = web_empowered_results_RAG(response, web_results)
        response_data = {"answer": output}
        return JSONResponse(response_data)
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return JSONResponse(content={"error": str(e)})

## Diagnosis APIs

@app.post("/get_diagnosis_CXR")
async def get_CXR(
    frontal_image: UploadFile = File(...),  # User's frontal chest X-ray
    lateral_image: Optional[UploadFile] = File(None),  # Optional lateral chest X-ray
    indication: str = Form(...),  # Clinical indication text
    technique: str = Form(...),  # Technique section
    comparison: Optional[str] = Form(None),  # Comparison section
    phrase: Optional[str] = Form(None),  # Optional phrase for phrase grounding
    query: Optional[str] = Form(None)  # Optional query by user
):
    try:
        # Read frontal image
        frontal_image_data = await frontal_image.read()

        # Check if lateral image was provided, else set it to None
        lateral_image_data = None
        if lateral_image and lateral_image.filename != "" and lateral_image!="":
            lateral_image_data = await lateral_image.read()

        # Convert image files into byte streams for the `get_diagnosis_CXR` function
        diagnosis_report = get_diagnosis_CXR(
            frontal_image_data=frontal_image_data,
            lateral_image_data=lateral_image_data,
            indication=indication,
            technique=technique,
            comparison=comparison,
            phrase="Pleural effusion" if phrase is None else phrase
        )

        cxr_data = {
            "indication": indication,
            "technique": technique,
            "comparison": comparison,
            "phrase": phrase  # This will be None if not provided
        }

        # Pass the byte data of images instead of `UploadFile` objects
        openai_diagnosis = process_image_query_and_json(
            image_data=frontal_image_data,  # Pass image byte stream
            optional_image_data=lateral_image_data,  # Pass lateral image byte stream, if present
            text_query=query,
            json_parameters=cxr_data
        )

        med_llm_results = get_medLLM_response(diagnosis_report)

        final_results = compare_responses(query=query, response1=openai_diagnosis, response2=med_llm_results, extra_info=cxr_data)
        seach_query = web_query(final_results)
        web_results = web_search(seach_query) 
        output = web_empowered_results(final_results, web_results)
        
        return JSONResponse(content=output)
    
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return JSONResponse(content={"error": str(e)})

@app.post("/get_diagnosis_dermat/")
async def get_dermat(
    images: Optional[List[UploadFile]] = None,
    query: Optional[str] = Form(None)
):
    # Store image files for the POST request
    image_files = []

    # Store paths to save images locally
    image_paths = []

    if images:
        # Read each image file into memory and append to the image_files list
        for image in images:
            image_content = await image.read()  # Read image content
            image_io = io.BytesIO(image_content)  # Create an in-memory binary stream
            image_files.append(('images', ('image.jpg', image_io, 'image/jpeg')))  # Append as "images"

            # Save the image to a directory (if needed)
            image_path = f"./images/{image.filename}"
            with open(image_path, "wb") as f:
                f.write(image_content)
            image_paths.append(image_path)

    try:
        # Assume these functions are defined elsewhere and handle the processing logic
        diagnosis_report = get_diagnosis_dermat(image_files=image_files, query=query)
        openAI_diagnosis = process_image_and_query(image_paths=image_paths, text_query=query)
        med_llm_results = get_medLLM_response(diagnosis_report)

        final_results = compare_responses_dermat(query=query, response1=openAI_diagnosis, response2=med_llm_results)
        search_query = web_query(final_results)
        web_results = web_search(search_query)
        output = web_empowered_results(final_results, web_results)

        return JSONResponse(content=output)
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return JSONResponse(content={"error": str(e)})


@app.post("/get_diagnosis_general")
async def get_general(
    image: UploadFile = File(...),
    query: str = Form(None)
):
    contents = await image.read()

    #save the file
    with open(f"{IMAGEDIR}{image.filename}", "wb") as f:
        f.write(contents)
    image_path = f"{IMAGEDIR}{image.filename}"
    print(image_path)
    try:
        openAI_diagnosis = process_image_and_query(image_paths = [image_path], text_query = query)
        med_llm_results = get_medLLM_response_general(openAI_diagnosis)
        final_results = compare_responses_general(query = query, response1 = openAI_diagnosis, response2 = med_llm_results)
        seach_query = web_query(final_results)
        web_results = web_search(seach_query)
        output = web_empowered_results(final_results, web_results)
        return JSONResponse(content=output)
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return JSONResponse(content={"error": str(e)})
    

# GAN APIs
# Directory to save generated CSV files
output_dir_tabular = r'A:\Downloads\Remedy.ai\generated_tabular_data'
os.makedirs(output_dir_tabular, exist_ok=True)

def save_dataframe_to_csv(dataframe, filename):
    """Save a DataFrame to a CSV file and return the file path."""
    file_path = os.path.join(output_dir_tabular, filename)
    dataframe.to_csv(file_path, index=False)
    return file_path

class BreastDataRequest(BaseModel):
    num_rows: int
class HeartDataRequest(BaseModel):
    num_rows: int
class DiabetesDataRequest(BaseModel):
    num_rows: int

@app.post("/generate_breast_data/")
async def generate_breast_data(num_rows: BreastDataRequest):
    try:
        # Generate breast data
        breast_data = breast_GAN(num_rows.num_rows)
        
        # Save to CSV
        file_path = save_dataframe_to_csv(breast_data, 'generated_breast_data.csv')
        
        # Return the CSV file as a response
        return FileResponse(file_path, media_type='text/csv', filename='generated_breast_data.csv')

    except Exception as e:
        # Handle any errors during the generation process
        raise HTTPException(status_code=500, detail=f"Breast GAN: An error occurred during generation: {str(e)}")

@app.post("/generate_diabetes_data/")
async def generate_diabetes_data(num_rows: DiabetesDataRequest):
    try:
        # Generate diabetes data
        diabetes_data = diabetes_GAN(num_rows.num_rows)
        
        # Save to CSV
        file_path = save_dataframe_to_csv(diabetes_data, 'generated_diabetes_data.csv')
        
        # Return the CSV file as a response
        return FileResponse(file_path, media_type='text/csv', filename='generated_diabetes_data.csv')

    except Exception as e:
        # Handle any errors during the generation process
        raise HTTPException(status_code=500, detail=f"Diabetes GAN: An error occurred during generation: {str(e)}")

@app.post("/generate_heart_data/")
async def generate_heart_data(num_rows: HeartDataRequest):
    try:
        # Generate heart data
        heart_data = heart_GAN(num_rows.num_rows)
        
        # Save to CSV
        file_path = save_dataframe_to_csv(heart_data, 'generated_heart_data.csv')
        
        # Return the CSV file as a response
        return FileResponse(file_path, media_type='text/csv', filename='generated_heart_data.csv')

    except Exception as e:
        # Handle any errors during the generation process
        raise HTTPException(status_code=500, detail=f"Heart GAN: An error occurred during generation: {str(e)}")


class KneeDataRequest(BaseModel):
    num_images: int

class ChestDataRequest(BaseModel):
    num_images: int

# GAN Images API
@app.post("/generate_chest_images/")
async def generate_chest_images(num_images: KneeDataRequest):
    try:
        # Call the generate_images_gan_chest function
        zip_file_path = generate_images_gan_chest(num_images.num_images)

        # Return the zip file as a response
        return FileResponse(zip_file_path, media_type='application/zip', filename='generated_chest_images.zip')

    except Exception as e:
        # Handle any errors during the generation process
        raise HTTPException(status_code=500, detail=f"Chest GAN: An error occurred during generation: {str(e)}")

@app.post("/generate_knee_images/")
async def generate_knee_images(num_images: KneeDataRequest):
    try:
        # Call the generate_images_gan_knee function
        zip_file_path = generate_images_gan_knee(num_images.num_images)

        # Return the zip file as a response
        return FileResponse(zip_file_path, media_type='application/zip', filename='generated_knee_images.zip')

    except Exception as e:
        # Handle any errors during the generation process
        raise HTTPException(status_code=500, detail=f"Knee GAN: An error occurred during generation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("application:app", host="localhost", port=8080, reload=True)