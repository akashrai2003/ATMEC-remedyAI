import requests
import os
import io
import openai
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import base64
import json
from io import BytesIO
# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_TYPE = os.getenv("AZURE_OPENAI_API_TYPE")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # Update the environment variable name
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

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

# URL of your FastAPI server where /generate_report is running
fastapi_url_cxr = "https://d64e-34-124-230-180.ngrok-free.app/generate_report"
fastapi_url_derma = "https://4f7c-34-123-16-117.ngrok-free.app/process"
fastapi_url_medllm = "https://1d36-34-126-136-138.ngrok-free.app/get_diagnosis/"
fastapi_url_medllm_general = "https://1d36-34-126-136-138.ngrok-free.app/get_diagnosis/"

def get_diagnosis_CXR(frontal_image_data, lateral_image_data=None, indication="", technique="", comparison="", phrase=None):
    # Prepare files as byte streams
    files = {
        "frontal_image": ("frontal_xray.png", BytesIO(frontal_image_data), "image/png")
    }

    if lateral_image_data:
        files["lateral_image"] = ("lateral_xray.png", BytesIO(lateral_image_data), "image/png")

    # Prepare form data
    data = {
        "indication": indication,
        "technique": technique,
        "comparison": comparison
    }

    if phrase:
        data["phrase"] = phrase

    # Send POST request to FastAPI server
    response = requests.post(fastapi_url_cxr, files=files, data=data)

    # Check the response status
    if response.status_code == 200:
        # Get and return the diagnosis report
        report = response.json().get("report")
        print("Received report:", report)
        return report
    else:
        print("Error:", response.status_code, response.text)
        return None

def get_diagnosis_dermat(image_files, query):
    # Ensure the images are passed under the correct form field name `images`
    data = {"text": query}  # Use the correct key for your text input
    
    # Send the POST request to the FastAPI server
    response = requests.post(
        fastapi_url_derma, 
        files=image_files, 
        data=data
    )
    
    if response.status_code == 200:
        # Get the JSON response with the model output
        model_response = response.json().get("output")
        print("Model Response:", model_response)
        return model_response
    else:
        print("Error:", response.status_code, response.text)
        return None

def get_medLLM_response(query):
    # Make the POST request to the FastAPI server
    response = requests.post(fastapi_url_medllm, json={"query": query})
    
    # Check if the request was successful
    if response.status_code == 200:
        # Get the JSON response with the answer
        answer = response.json().get("answer")
        print("Received answer:", answer)
        return answer
    else:
        print("Error:", response.status_code, response.text)
        return None

def get_medLLM_response_general(query):
    # Make the POST request to the FastAPI server with form data
    response = requests.post(fastapi_url_medllm_general, data={"query": query})
    
    # Check if the request was successful
    if response.status_code == 200:
        # Get the JSON response with the answer
        answer = response.json().get("response")
        print("Received answer:", answer)
        return answer
    else:
        print("Error:", response.status_code, response.text)
        return None
def process_image_and_query(image_paths, text_query):
    try:
        # Prepare the message list, starting with the system prompt
        messages = []
        messages.append(
            SystemMessage(
                content=[
                    {"type": "text", "text": "You are a medical expert. Based on the image(s) and text query, provide a detailed diagnosis."},
                ]
            )
        )

        # Initialize a list to hold the content (text + images)
        content_list = [{"type": "text", "text": text_query}]

        # Loop through the list of image paths and add each image as base64
        for image_path in image_paths:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                # Append the image data to the content list
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                })

        # Add the user message with the text query and images
        messages.append(HumanMessage(content=content_list))

        # Send the messages using AzureChatOpenAI or any LLM service you're using
        response = llmGetResponse.invoke(messages)

        # Output the response from the model
        print(f"Response: {response.content}")
        return response.content

    except Exception as e:
        print(f"Error processing the image and query: {e}")
        return None



def process_image_query_and_json(image_data, text_query=None, optional_image_data=None, json_parameters=None):
    try:
        # Convert the mandatory image to base64 (this assumes 'image_data' is a byte stream)
        image_data_base64 = base64.b64encode(image_data).decode("utf-8")

        # Prepare the message list, starting with the system prompt
        messages = [
            SystemMessage(
                content=[
                    {"type": "text", "text": "You are a medical expert. Based on the input images, text query, and parameters, provide a detailed diagnosis."},
                ]
            )
        ]

        # Add the primary image and text query to the message content
        content = [
            {"type": "text", "text": text_query if text_query else "No query provided"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data_base64}"},
            },
        ]

        # If an optional image is provided, add it to the message content
        if optional_image_data:
            optional_image_data_base64 = base64.b64encode(optional_image_data).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{optional_image_data_base64}"},
                }
            )

        # If JSON parameters are provided, include them as well
        if json_parameters:
            json_str = json.dumps(json_parameters)  # Convert the JSON object to a string
            content.append(
                {
                    "type": "text",
                    "text": f"Parameters: {json_str}",
                }
            )

        # Add the user message to the messages list
        messages.append(HumanMessage(content=content))

        # Send the messages using AzureChatOpenAI (or any other LLM)
        response = llmGetResponse.invoke(messages)

        # Output the response from the model
        print(f"Response: {response.content}")
        return response.content
    except Exception as e:
        print(f"Error processing the images, query, and parameters: {e}")
        return None

def compare_responses_dermat(query, response1, response2):
    try:
        # Prepare the comparison prompt
        prompt_template = """
        You are a medical expert evaluating two responses. 
        Give me a combined diagnosis from the 2 responses from different LLMs believing that the second response is giving out info from the image and the 1st one is more readable and describing:
        Query: {query}
        Response 1: {response1}
        Response 2: {response2}

        """

        # Fill the prompt with the actual responses
        prompt = prompt_template.format(query=query, response1=response1, response2=response2)

        # Prepare messages for the model
        messages = [
            SystemMessage(
                content=[
                    {"type": "text", "text": "You are an expert system capable of comparing responses and combining into a better response which is more reliable."}
                ]
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt}
                ]
            )
        ]

        # Invoke the model with the prepared messages
        response = llmGetResponse.invoke(messages)

        # Output the response from the model
        print(f"Comparison Result: {response.content}")
        return response.content
    except Exception as e:
        print(f"Error comparing responses: {e}")
        return None
    
def compare_responses(query, response1, response2, extra_info = None):
    try:
        # Prepare the comparison prompt
        prompt_template = """
        You are a medical expert evaluating two responses. 
        Give me a combined diagnosis from the 2 responses from different LLMs believing that the second response is a bit more accurate and the 1st one is more readable and describing:
        Query: {query}
        Extra Info: {extra_info}
        Response 1: {response1}
        Response 2: {response2}

        """

        # Fill the prompt with the actual responses
        prompt = prompt_template.format(query=query, response1=response1, response2=response2, extra_info=extra_info)

        # Prepare messages for the model
        messages = [
            SystemMessage(
                content=[
                    {"type": "text", "text": "You are an expert system capable of comparing responses and combining into a better response which is more reliable."}
                ]
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt}
                ]
            )
        ]

        # Invoke the model with the prepared messages
        response = llmGetResponse.invoke(messages)

        # Output the response from the model
        print(f"Comparison Result: {response.content}")
        return response.content
    except Exception as e:
        print(f"Error comparing responses: {e}")
        return None
    
def compare_responses_general(query, response1, response2):
    try:
        # Prepare the comparison prompt
        prompt_template = """
        You are a medical expert evaluating two responses. 
        Give me a combined diagnosis from the 2 responses from different LLMs believing that the second response is more technical in terms and the 1st one is more readable and describing it in detail:
        Query: {query}
        Response 1: {response1}
        Response 2: {response2}

        """

        # Fill the prompt with the actual responses
        prompt = prompt_template.format(query=query, response1=response1, response2=response2)

        # Prepare messages for the model
        messages = [
            SystemMessage(
                content=[
                    {"type": "text", "text": "You are an expert system capable of comparing responses and combining into a better response which is more reliable."}
                ]
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt}
                ]
            )
        ]

        # Invoke the model with the prepared messages
        response = llmGetResponse.invoke(messages)

        # Output the response from the model
        print(f"Comparison Result: {response.content}")
        return response.content
    except Exception as e:
        print(f"Error comparing responses: {e}")
        return None

def web_query(query):
    try:
        prompt_template = '''
            Based on the response given below by the medical expert, please give me a query that I can run on the web to get more information about the diagnosis while also getting treatment options and other information useful for patients.
            Response: {query}'''
        # Fill the prompt with the actual responses
        prompt = prompt_template.format(query=query)

        # Prepare messages for the model
        messages = [
            SystemMessage(
                content=[
                    {"type": "text", "text": "You are an expert system capable of giving accurate and comapct queries for web searches under 350 characters to get information from the web such as treatments, better diagnoses and other relevant info."}
                ]
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt}
                ]
            )
        ]

        # Invoke the model with the prepared messages
        response = llmGetResponse.invoke(messages)

        # Output the response from the model
        print(f"Comparison Result: {response.content}")
        return response.content
    except Exception as e:
        print(f"Error comparing responses: {e}")
        return None
    
def web_empowered_results(response, web_results):
    try:
        prompt_template = '''
            Based on the response given below by the medical expert and the web search results which are more recent and could be accurate too, please give me a detailed report that combines the information from both sources to provide a comprehensive overview of the diagnosis, treatment options, and other relevant information.
            Medical Expert Response: {response}
            Web Search Results: {web_results}'''
        # Fill the prompt with the actual responses
        prompt = prompt_template.format(response=response, web_results=web_results)

        # Prepare messages for the model
        messages = [
            SystemMessage(
                content=[
                    {"type": "text", "text": "You are an expert system capable of combining information from medical experts and web searches to provide a detailed report on the diagnosis, treatment options, and other relevant information."}
                ]
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt}
                ]
            )
        ]

        # Invoke the model with the prepared messages
        response = llmGetResponse.invoke(messages)

        # Output the response from the model
        print(f"Comparison Result: {response.content}")
        return response.content
    except Exception as e:
        print(f"Error comparing responses: {e}")
        return None
    

def web_query_RAG(query):
    try:
        prompt_template = '''
            Based on the response given below by the medical expert, please give me a query that I can run on the web to get more information about the diet plan and schedule to be followed by the patient based on the his medical records.
            Response: {query}'''
        # Fill the prompt with the actual responses
        prompt = prompt_template.format(query=query)

        # Prepare messages for the model
        messages = [
            SystemMessage(
                content=[
                    {"type": "text", "text": "You are an expert system capable of giving accurate and comapct queries for web searches under 350 characters to get information from the web about diet and nutrition based on the patients health records."}
                ]
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt}
                ]
            )
        ]

        # Invoke the model with the prepared messages
        response = llmGetResponse.invoke(messages)

        # Output the response from the model
        print(f"Comparison Result: {response.content}")
        return response.content
    except Exception as e:
        print(f"Error comparing responses: {e}")
        return None
    
def web_empowered_results_RAG(response, web_results):
    try:
        prompt_template = '''
            Based on the response given below by the medical expert and the web search results which are more recent and could be accurate too, please give me the best diet plan and what to avoid eating as well as what to prefer eating.
            Medical Expert Response: {response}
            Web Search Results: {web_results}'''
        # Fill the prompt with the actual responses
        prompt = prompt_template.format(response=response, web_results=web_results)

        # Prepare messages for the model
        messages = [
            SystemMessage(
                content=[
                    {"type": "text", "text": "You are an expert system capable of combining information from medical experts and web searches to provide the perfect diet plan with all important aspects of eating according to the patients medical record."}
                ]
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt}
                ]
            )
        ]

        # Invoke the model with the prepared messages
        response = llmGetResponse.invoke(messages)

        # Output the response from the model
        print(f"Comparison Result: {response.content}")
        return response.content
    except Exception as e:
        print(f"Error comparing responses: {e}")
        return None
    

# # Example usage
# if __name__ == "__main__":
#     frontal_image_path = "path_to_frontal_xray.png"
#     lateral_image_path = "path_to_lateral_xray.png"  # Optional
#     indication = "Dyspnea"
#     technique = "PA and lateral views of the chest"
#     comparison = "None"
    
#     # Get the diagnosis (this will send a request to your FastAPI server)
#     diagnosis_report = get_diagnosis_CXR(
#         frontal_image_path,
#         lateral_image_path=lateral_image_path,
#         indication=indication,
#         technique=technique,
#         comparison=comparison,
#         phrase="Pleural effusion"  # Optional
#     )
    
#     # You can now use diagnosis_report as needed in this server
