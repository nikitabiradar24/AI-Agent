from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.utilities import WikipediaAPIWrapper, SerpAPIWrapper
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# LLM (OpenRouter)
llm = ChatOpenAI(
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    model="gpt-3.5-turbo"
)

# Tools
wiki = WikipediaAPIWrapper()
search = SerpAPIWrapper()

tools = [
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Useful for general knowledge"
    ),
    Tool(
        name="Google Search",
        func=search.run,
        description="Useful for latest information"
    )
]

# Agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True #show step-by-step thinking
)

# Run loop
while True:
    query = input("\nAsk something (type 'exit'): ")

    if query.lower() == "exit":
        break

    try:
        response = agent.run(query)
        print("\nAnswer:\n", response)
    except Exception as e:
        print("\nError:", str(e))