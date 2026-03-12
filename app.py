from smolagents import CodeAgent, load_tool, tool
import datetime
import requests
import pytz
import yaml
from smolagents.models import LiteLLMModel
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool
from tools.web_search import DuckDuckGoSearchTool

@tool
def get_current_time(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()
visit_webpage = VisitWebpageTool()
web_search = DuckDuckGoSearchTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud'

model = LiteLLMModel(
max_tokens=2096,
temperature=0.3,
model_id="ollama/llama3:8b",# it is possible that this model may be overloaded
custom_role_conversions=None,
)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)



agent = CodeAgent(
    model=model,
    tools=[visit_webpage, web_search, final_answer], ## add your tools here (don't remove final answer)
    max_steps=9,
    verbosity_level=1,
    grammar=None,
    planning_interval=2,
    name=None,
    description="Agent that searches the web and summarizes reliable information.",
    prompt_templates=prompt_templates
)


def run_agent(query: str) -> str:
    return agent.run(query, stream=False)


'''if __name__ == "__main__":
    # Simple manual test
    print(agent.run("In a 1979 interview, Stanislaus Ulam discusses with Martin Sherwin about other great physicists of his time, including Oppenheimer.  What does he say was the consequence of Einstein learning too much math on his creativity, in one word?"))'''