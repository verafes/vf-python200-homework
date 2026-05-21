# warmup_07.py
import os

from openai import OpenAI
from datetime import datetime
import json
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR / "resources"

load_dotenv()
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

client = OpenAI()
print('OpenAI client created.')


# --- Lesson 02: Tool Definitions and the ReAct Loop ---

# Q1 — Define celsius_to_fahrenheit + JSON schema + direct calls
print("\n--- Q1: Celsius to Fahrenheit ---")

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

# JSON schema describing this function (like get_current_time in lesson)
celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "Temperature in Celsius"
                }
            },
            "required": ["celsius"]
        }
    }
}

# Direct calls (not through an agent)
print(f"Q1 Output 1: {celsius_to_fahrenheit(0)}")
print(f"Q1 Output 2: {celsius_to_fahrenheit(100)}")
print(f"Q1 Output 3: {celsius_to_fahrenheit(-40)}")


# Q2 — Copy run_agent from lesson (only get_current_time tool)
print("\n--- Q2: Copy run_agent ---")

# Prediction before running:
# 1. Will run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call?
# No. The only tool available is get_current_time, and the query is not about time.
# The model will answer directly with text.

# 2. How many API calls will be made?
# Exactly ONE API call. The model receives the prompt, answers directly.
# The agent loop ends immediately because no tool call is requested.

# Tool: get_current_time (from lesson)
def get_current_time():
    return datetime.now().isoformat()

get_current_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current time in ISO format.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

tools = [get_current_time_schema]


def run_agent(prompt: str):
    """
    Simple‑tool ReAct agent for a single user prompt.
    Sends prompt, runs tool if requested, returns final answer.
    """
    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                         Use the tool get_current_time whenever a user asks about the time.'''

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt},
    ]

    # first API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    # Record model responses
    msg = response.choices[0].message
    print("First response received from model...")
    if msg.content:
        print("Assistant text:", msg.content)
    else:
        print("Assistant requested a tool call.")

    # check if the model wants to call a tool
    if msg.tool_calls:
        print("Agentic mode engaged...")
        tool_call = msg.tool_calls[0]
        function_name = tool_call.function.name
        if function_name == "get_current_time":
            tool_result = get_current_time()
        else:
            tool_result = f'Error: unknown tool {function_name}.'

        print(f"Tool called: {function_name}")
        print(f"Tool result: {tool_result}")

        messages.append(msg)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": tool_result
        })
        followup_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        print("Second response received from model...")
        print(followup_response)

        return followup_response.choices[0].message.content or ""
    else:
        print("\nNo tools needed....")

    # Otherwise return the model's direct answer
    return msg.content or ""


# Run Q2 test
q2_result = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print(f"\nQ2 Result (Simple‑tool ReAct agent test): {q2_result}")

# My prediction was correct.
# No tool was called — the model answered directly because only had access to get_current_time tool.


# Q3 — Extend agent to support BOTH tools
print("\n--- Q3: Extend agent ---")

tools_extended = [
    get_current_time_schema,
    celsius_to_fahrenheit_schema
]

def run_agent_extended(prompt: str):
    """
    Multi‑tool ReAct agent (time + Celsius)
    Lets model choose a tool, runs it, returns final answer.
    """
    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                             Use the tool get_current_time whenever a user asks about the time.'''

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools_extended,
        tool_choice="auto"
    )

    msg = response.choices[0].message
    print("First response received from model...")

    if msg.content:
        print("Assistant text:", msg.content)
    else:
        print("Assistant requested a tool call.")

    # If tool call requested
    if msg.tool_calls:
        print("Agentic mode engaged...")
        tool_call = msg.tool_calls[0]
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if function_name == "celsius_to_fahrenheit":
            tool_result = celsius_to_fahrenheit(args["celsius"])

        elif function_name == "get_current_time":
            tool_result = get_current_time()

        messages.append(msg)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": tool_result
        })

        followup_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        print("Second response received from model...")
        return followup_response.choices[0].message.content or ""
    else:
        print("\nNo tools needed....")

    # Otherwise return direct answer
    return msg.content or ""


# Run Q3 tests
prompt_a = "What is 37 degrees Celsius in Fahrenheit?"
print(f"Test 1, prompt A: {prompt_a}")
response_a = run_agent_extended(prompt_a)
print("Response A:", response_a)

# A tool WAS called — the model recognized a temperature conversion request.

prompt_b = "What is the boiling point of water in plain English?"
print(f"\nTest 2, prompt B: {prompt_b}")
response_b = run_agent_extended(prompt_b)
print("\nResponse B:", response_b)

# No tool was called — this is a knowledge question, not direct temperature conversion request.
# The LLM can answer this from its own knowledge without using a tool.


# --- Lesson 03: Multi-Tool Agent ---

# Q4 — Add compute_correlation to CsvManager

class CsvManager:
    """Manage CSV loading and simple data operations."""
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers ---
    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) -----
    def list_csv_files(self):
        """ List available CSV files in resources/."""
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """ Load a CSV file from resources/ and make it the active dataset."""
        filename = self._normalize_csv_name(filename)
        file_path = self.resources_dir / filename

        if not file_path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(file_path)
        self.csv_name = filename

        return {
            "message": f"Loaded {file_path} with shape {self.df.shape}.",
            "columns": list(self.df.columns)}

    def get_columns(self):
        """ Return column names for the currently loaded CSV. """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """ Simple summary for a single column using pandas.describe(). """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.

        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."

        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"

        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None

        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."

        title_csv = self.csv_name or "current CSV"

        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."

        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"

        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()
        return f"Plotted {y} vs {x} as a {plot_type}."

    # new Q4 tool
    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        # Check if CSV is loaded
        if self.df is None:
            return {"error": "No CSV loaded."}

        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {"error": f"One or both columns not found: {col1}, {col2}"}

        try:
            r, p = pearsonr(self.df[col1], self.df[col2])

            return {
                "col1": col1,
                "col2": col2,
                "pearson_r": round(float(r), 4),
                "p_value": round(float(p), 4),
            }
        except Exception as e:
            return {"error": str(e)}

# Create CsvManager Instance
csv_manager = CsvManager(RESOURCES_DIR)

# Add the tool to node_tools
node_tools = {
    "list_csv_files": csv_manager.list_csv_files,
    "load_csv": csv_manager.load_csv,
    "get_columns": csv_manager.get_columns,
    "summarize_columns": csv_manager.summarize_columns,
    "describe_column": csv_manager.describe_column,
    "plot_data": csv_manager.plot_data,
    "compute_correlation": csv_manager.compute_correlation,
}

# Add the correlation tool to tools_schema
tools_schema = [
    # list csv files
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    # load csv
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"]
            }
        }
    },
    # get columns
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV"
        }
    },

    # summarize columns
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Summarize columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                }
            }
        }
    },
    # describe column
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Describe one column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"]
            }
        }
    },
    # plot data
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"]
            }
        }
    },
    # tool - compute_correlation schema
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute Pearson correlation between two numeric columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {"type": "string", "description": "First column name"},
                    "col2": {"type": "string", "description": "Second column name"}
                },
                "required": ["col1", "col2"]
            }
        }
    }
]

# The run_agent_cycle method is defined as a single ReAct loop to generate the response to a user query.
def run_agent_cycle(messages, user_input, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt,
    and then user text will be appended.

    The loop has three main steps:

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.

    ACT:
      - If tools are requested, run the Python functions

    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the
        final answer for this react cycle, this implies that reasoning alone without
        tool calls was enough.
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_input})

    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the LLMs conversation history.
        The model will read this tool output on the next REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,}
        return tool_message

    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message


        # store assistant messages as plain dicts
        assistant_entry = {"role": "assistant", "content": msg.content}
        # include tool_calls in dict form
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        # append the assistant message to the conversation history.
        messages.append(assistant_entry)

        # stop condition: no tool calls means the model is answering directly.
        if not msg.tool_calls:
            return msg.content

        # ACT + OBSERVE: run each tool call, then append its result.
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}

            # OBSERVE: append the tool result back into the conversation history.
            messages.append(observe_tool_result(tool_call.id, result))

            # After appending info about all tool outputs, it loops back and REASON again.

    return "I hit the tool-round limit. Try a simpler request."

SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)


# Q5: run agent cycle and print result
print("\n--- Q5 ---")

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]
file = "bike_commute.csv"
csv_path = RESOURCES_DIR / file

result = run_agent_cycle(
    messages,
    f"Load {file} and compute the correlation between avg_traffic_density and avg_speed_kmh."
)

print("Final Agent Response:")
print(result)


# Q6 — Print full message history
print("\n--- Q6 ---")

# ReAct Loop Roles
# system: contains instructions that define the agent behavior.
# user: the user's request or question.
# assistant: the LLM reasoning step. The assistant decides whether to answer directly or call tools.
# tool: the actual output returned from a tool execution ack to the model.
# The assistant observes this result and continues reasoning using the new information.

# helpers to clean up the messages list
def normalize_react_messages(messages):
    cleaned = []
    for m in messages:
        m2 = dict(m)
        if m2.get("role") == "tool":
            try:
                m2["content"] = json.loads(m2["content"])
            except Exception:
                pass
        cleaned.append(m2)
    return cleaned

def unescape_newlines(json_text: str) -> str:
    return json_text.replace("\\n", "\n")

pretty_output = json.dumps(normalize_react_messages(messages), indent=2)
pretty_output = unescape_newlines(pretty_output)
print(pretty_output)


# --- Lesson 04: smolagents ---
# Q7 - wrapping function as a smolagents tool
print("\n--- Q7 ---")

@tool
def list_csv_files() -> dict:
    """ List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()

@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)

@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()

@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns).
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_manager.summarize_columns(columns)

@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_manager.describe_column(column)

@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """"Plot from the active CSV.

    Args:
        y: Column name to plot on the y-axis.
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.

    Returns:
        Generates and shows the plot.
        Retirms a short success message string, or an error dict/string..
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute Pearson correlation between two columns.

    Args:
        col1: Name of the first column.
        col2: Name of the second column.

    Returns:
        A dictionary with correlation results.
    """
    return csv_manager.compute_correlation(col1, col2)

print(compute_correlation.description)

# smolagents automatically generates a tool description from:
# - the function name
# - the type hints
# - the docstring

# In Q4, I had to manually write a JSON schema with:
# - name
# - description
# - parameters
# - required fields

# smolagents only needs type hints + docstring to build the schema.
# The developer must provide clear type hints and a good docstring.


# Q8 — Build ToolCallingAgent + CodeAgent
print("\n--- Q8 ---")

TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]

# build model
model_to_use = "gpt-4o-mini"
model = OpenAIServerModel(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_id=model_to_use,
)

CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

You can do two kinds of actions:
1) Call the provided tools.
2) Write and execute Python code when tools are not enough.

Rules:
- Prefer tools for simple tasks.
- IMPORTANT: If the user requests plot styling (color, marker, title text, labels, grid, etc.)
  that the plot_data tool cannot control, DO NOT call plot_data.
  Instead, write matplotlib code directly so the plot matches the request.
  If code execution fails, do not fall back to plot_data when the user requested styling (like color). 
  Explain what failed and what you would need to proceed.
- Be honest: only claim you did something if the code or tool actually did it.
- Assume the active dataset lives in csv_manager.df after a CSV is loaded.
"""

# build both agents
tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model,
    instructions=SYSTEM_PROMPT,
)

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Run the prompt through both
prompt = ("Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots. "
          "Save the plot into the outputs/ folder.")

response_tool = tool_agent.run(prompt)
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})

print("ToolCallingAgent:", response_tool)
print("CodeAgent:", response_code)


# Q8 Analysis:
# - The ToolCallingAgent cannot change dot color because plot_data does not support color.
#   It will call plot_data(...) and then hallucinate that it made green dots.

# - The CodeAgent writes custom matplotlib code, so it CAN change the dot color.
#   It produces an actual scatter plot with green dots.

# - ToolCallingAgent is good when tools are sufficient and safe.
# - CodeAgent is better when the user needs custom logic or plot styling.


# --- Q9 ---
# 1. A task where a ToolCallingAgent is a better choice:
# ToolCallingAgent is best when the task should stay safe, predictable, and tightly controlled.
# For example, loading a CSV, listing and summarizing columns, computing correlation, or creating a basic scatter plot
# using the predefined tools. These tasks are a good fit because the agent is restricted to a fixed set
# of operations, so cannot invent new behavior -- it can only call the tools I explicitly exposed.
# This guarantees consistency and the agent from doing anything outside the intended workflow.
#
# 2. A meaningful risk of using a CodeAgent:
# CodeAgent can generate and execute arbitrary Python code.
# This means it could accidentally write unsafe code, overwrite files, create infinite loops,
# or run plotting code that behaves unpredictably (as I saw when it tried to show GUI plots).
# The ToolCallingAgent never has this risk because it cannot execute free‑form code --
# it can only call safe, predefined tools.