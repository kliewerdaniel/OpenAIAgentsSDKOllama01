from ollama_client import OllamaClient
from openai.types.chat import ChatCompletion, ChatCompletionMessage
import agents.agent as agent_module
from agents.agent import Agent
from agents.run import Runner, RunConfig
from agents.models import _openai_shared
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set placeholder OpenAI API key to avoid initialization errors
_openai_shared.set_default_openai_key("placeholder-key")

# Create message types mimicking what the Agent expects
class HumanMessage:
    def __init__(self, content):
        self.content = content
        self.role = "user"

class SystemMessage:
    def __init__(self, content):
        self.content = content
        self.role = "system"

# Store original init for Agent class
original_init = Agent.__init__

def patched_init(self, *args, **kwargs):
    """Replace the model with OllamaClient if not provided."""
    if "model" not in kwargs:
        kwargs["model"] = OllamaClient(model_name="mistral")
    original_init(self, *args, **kwargs)

# Apply the patched init
Agent.__init__ = patched_init


# Class for a structured tool call
class ToolCall:
    def __init__(self, name, inputs=None):
        self.name = name
        self.inputs = inputs or {}

# Define a response class that matches what main.py expects
class AgentResponse:
    def __init__(self, result):
        # Extract the message from the final output
        if hasattr(result, 'final_output'):
            if isinstance(result.final_output, str):
                self.message = result.final_output
            else:
                self.message = str(result.final_output)
        else:
            self.message = "I'm sorry, I couldn't process that request."
        
        # Get conversation ID if available
        self.conversation_id = getattr(result, 'conversation_id', None)
        
        # Initialize tool_calls
        self.tool_calls = []
        
        # Extract tool calls from raw_responses
        if hasattr(result, 'raw_responses'):
            for response in result.raw_responses:
                try:
                    if hasattr(response, 'output') and hasattr(response.output, 'tool_calls'):
                        for tool_call in response.output.tool_calls:
                            # Handle the case where tool_call is a dict
                            if isinstance(tool_call, dict):
                                name = tool_call.get('name', 'unknown_tool')
                                inputs = tool_call.get('inputs', {})
                                self.tool_calls.append(ToolCall(name, inputs))
                            else:
                                # Assume it's already an object with name and inputs attributes
                                self.tool_calls.append(tool_call)
                except Exception as e:
                    logger.error(f"Error extracting tool calls: {str(e)}")


# Add a run method to the Agent class
def run(self, message, conversation_id=None):
    """Run the agent with the given message.
    
    Args:
        message: The user message to process
        conversation_id: Optional conversation ID for continuity
        
    Returns:
        A response object with message, conversation_id, and tool_calls attributes
    """
    try:
        # Create a direct prompt for the model
        prompt = f"""
        {self.instructions}
        
        User query: {message}
        """
        
        # Get a response directly from the model (OllamaClient)
        response = self.model.chat.completions.create(
            model="mistral",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        
        # Extract the text response
        response_text = response.choices[0].message.content
        
        # Create a minimal result object with just the response text
        class MinimalResult:
            def __init__(self, text, conv_id):
                self.final_output = text
                self.conversation_id = conv_id
                self.raw_responses = []
        
        result = MinimalResult(response_text, conversation_id)
        
        # Return a response object
        return AgentResponse(result)
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error running agent: {str(e)}\n{error_traceback}")
        
        # Create a basic response with the error message
        response = AgentResponse(None)
        response.message = f"An error occurred: {str(e)}"
        return response


# Make sure the run method is applied to the Agent class
Agent.run = run

# Debugging statement - log when the adapter is loaded
print("Agent adapter loaded, Agent class patched with run method.")
