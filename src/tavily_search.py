import os
from tavily import TavilyClient
from langchain_openai import AzureChatOpenAI
from langchain.adapters.openai import convert_openai_messages

from dotenv import load_dotenv
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
API_KEY = os.getenv("TAVILY_API_KEY")
client = TavilyClient(api_key=API_KEY)

def web_search(query):
    content = client.search(query, search_depth="advanced")["results"]
    prompt = [{
        "role": "system",
        "content":  f'You are an AI critical thinker research assistant. '\
                    f'Your sole purpose is to write well written, critically acclaimed,'\
                    f'objective and structured reports on given text.'
    }, {
        "role": "user",
        "content": f'Information: """{content}"""\n\n' \
                f'Using the above information, answer the following'\
                f'query: "{query}" in a detailed report --'\
                f'Please use MLA format and markdown syntax.'
    }]

    # Step 4. Running OpenAI through Langchain
    lc_messages = convert_openai_messages(prompt)
    report = llmGetResponse.invoke(lc_messages).content
    return report
