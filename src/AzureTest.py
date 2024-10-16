import os
import openai
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import base64
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


# Function to test API key
def test_openai_gpt4():
    try:
        query = "What is Azure OpenAI API?"

        # Set up a simple prompt to test the API key
        prompt_template = """
        Question: {question}
        Please respond to this query: {question}
        """
        prompt = prompt_template.format(question=query)

        # Make the prediction request using LangChain's AzureChatOpenAI model
        prediction = llmGetResponse.invoke(prompt)
        print(f"Test query response: {prediction.content}")
    except Exception as e:
        print(f"Error testing the API key: {e}")

def process_image_and_query(image_path, text_query):
    try:
        # Read the image and convert it to base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        # Prepare the message list, starting with the system prompt if provided
        messages = []
        messages.append(
            SystemMessage(
                content=[
                    {"type": "text", "text": "You are a medical expert and based on the image and text query, provide a detailed diagnosis."},
                ]
            )
        )

        # Add the user message with the text query and image
        messages.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": text_query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            )
        )

        # Send the messages using AzureChatOpenAI
        response = llmGetResponse.invoke(messages)

        # Output the response from the model
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Error processing the image and query: {e}")


image_path = r"A:\Downloads\Remedy.ai\datasets\med.jpeg"  # Replace with the actual image file path
text_query = "Can you help me know if I do have a medical condition?"  # Replace with the text query
system_prompt = "You are an expert in analyzing medical images. Please describe the context in this image."

# Call the function with the image, text query, and system prompt
process_image_and_query(image_path, text_query)


# # Test the function
# test_openai_gpt4()
