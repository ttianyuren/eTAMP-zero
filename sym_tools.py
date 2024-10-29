import re
import contextlib, io
import pickle


def parse_action_string(action_str):
    """
    Extracts the action and parameters from the action string.
    Example: "(place o1 r3)" -> action: "place", parameters: ["o1", "r3"]
    """
    # Use regular expressions to extract the parts of the string
    match = re.match(r"\((\w+)\s+(.*)\)", action_str)

    if match:
        action = match.group(1)  # Extract the action
        params = match.group(2).split()  # Extract the parameters as a list
        return action, params
    else:
        return None, None


def call_without_printing(func, *args, **kwargs):
    # Redirect stdout to a dummy string stream
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def load_data(path_file):
    with open(path_file, 'rb') as f:
        x = pickle.load(f)

    return x


def save_data(data, path_file):
    with open(path_file, "wb") as f:
        pickle.dump(data, f)
    print("writen to",path_file)
