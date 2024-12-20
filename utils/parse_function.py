# # import json
# # import re

# # def extract_function_names(text):
# #     # Find JSON content within the text using regular expressions
# #     json_pattern = r'```(?:json)?(.*?)```'
# #     matches = re.findall(json_pattern, text, re.DOTALL)

# #     function_names = []

# #     for match in matches:
# #         try:
# #             # Parse the JSON content
# #             json_data = json.loads(match)
# #             # Iterate through the list of functions and extract the names
# #             for item in json_data:
# #                 function_name = item.get('function_name')
# #                 if function_name:
# #                     function_names.append(function_name)
# #         except json.JSONDecodeError as e:
# #             print(f"Error decoding function name JSON: {e}")

# #     return function_names


# # def extract_function_parameters(text):
# #     # Find JSON content within the text using regular expressions
# #     json_pattern = r'```(?:json)?(.*?)```'
# #     matches = re.findall(json_pattern, text, re.DOTALL)

# #     parameter_list = []

# #     for match in matches:
# #         try:
# #             # Parse the JSON content
# #             json_data = json.loads(match)
# #             # Iterate through the list of functions and extract the parameters
# #             for item in json_data:
# #                 function_parameters = item.get('function_parameters')
# #                 params = dict()
# #                 if function_parameters:
# #                     for key, value in function_parameters.items():
# #                       params[key] = value
# #                     parameter_list.append(params)
# #         except json.JSONDecodeError as e:
# #             print(f"Error decoding function parameters JSON: {e}")

# #     return parameter_list

# # def extract_iter(text):
# #     # Find JSON content within the text using regular expressions
# #     json_pattern = r'```(?:[^`]|`[^`]|``[^`])*```'
# #     matches = re.findall(json_pattern, text, re.DOTALL)

# #     ITER = False

# #     for match in matches:
# #         try:
# #             # Parse the JSON content
# #             json_data = json.loads(match)
# #             # Iterate through the list of functions and extract the parameters
# #             for item in json_data:
# #                 if "ITER" in item:  # Check if ITER key exists
# #                     iter_value = item["ITER"]
# #                     if isinstance(iter_value, bool):
# #                         ITER = iter_value
# #                     elif isinstance(iter_value, str) and iter_value.lower() == "false":
# #                         ITER = False
# #                     elif isinstance(iter_value, str) and iter_value.lower() == "true":
# #                         ITER = True
# #                     else:
# #                         print("ITER value is not boolean.")
# #                 else:
# #                     print("ITER key is missing in the JSON data.")
# #         except json.JSONDecodeError as e:
# #             print(f"Error decoding ITER JSON: {e}")

# #     return ITER

# # if __name__ == "__main__":
# #     # Example usage:
# #     text_output = """
# #      ```
# #     [
# #         {
# #             "function_name": "coder",
# #             "function_parameters": {
# #                 "query": "clone https://github.com/shankerabhigyan/dsa-code.git in './dsa-code'" 
# #             }, 
# #             "ITER": "False"
# #         }
# #     ]
# #     ```
# #     """

# #     function_names = extract_function_names(text_output)
# #     print(function_names)
# #     function_parameters = extract_function_parameters(text_output)
# #     print(function_parameters)
# #     iter = extract_iter(text_output)
# #     print(iter,type(iter))



# import json
# import re

# def extract_function_names(text):
#     json_pattern = r'```(?:json)?(.*?)```'
#     matches = re.findall(json_pattern, text, re.DOTALL)

#     function_names = []
#     for match in matches:
#         try:
#             json_data = json.loads(match)
#             for item in json_data:
#                 function_name = item.get('function_name')
#                 if function_name:
#                     function_names.append(function_name)
#         except json.JSONDecodeError as e:
#             print(f"Error decoding function name JSON: {e}")
#     return function_names


# def extract_function_parameters(text):
#     json_pattern = r'```(?:json)?(.*?)```'
#     matches = re.findall(json_pattern, text, re.DOTALL)

#     parameter_list = []
#     for match in matches:
#         try:
#             json_data = json.loads(match)
#             for item in json_data:
#                 function_parameters = item.get('function_parameters')
#                 if function_parameters:
#                     parameter_list.append(function_parameters)
#         except json.JSONDecodeError as e:
#             print(f"Error decoding function parameters JSON: {e}")
#     return parameter_list


# def extract_iter(text):
#     json_pattern = r'```(?:json)?(.*?)```'
#     matches = re.findall(json_pattern, text, re.DOTALL)

#     iter_value = False  # Default value
#     for match in matches:
#         try:
#             json_data = json.loads(match)
#             for item in json_data:
#                 if "ITER" in item:
#                     raw_iter = item["ITER"]
#                     # Convert ITER to a proper boolean
#                     if isinstance(raw_iter, bool):
#                         iter_value = raw_iter
#                     elif isinstance(raw_iter, str):
#                         iter_value = raw_iter.lower() == "true"
#                     else:
#                         print("ITER value is not boolean or string.")
#         except json.JSONDecodeError as e:
#             print(f"Error decoding ITER JSON: {e}")
#     return iter_value


# if __name__ == "__main__":
#     # Example usage:
#     text_output = """
#     ```json
# [
#     {
#         "function_name": "summary_text",
#         "function_parameters": {
#             "message": "Certainly! Here is a simple Python code to add two numbers:\n\n
# python\ndef add_two_numbers(a, b):\n    return a + b\n\n# Example usage:\nresult = add_two_numbers(3, 5)\nprint('The sum is:', result)\n
# "
#         }
#     }
# ]
#     ```
#     """

#     function_names = extract_function_names(text_output)
#     print("Function Names:", function_names)
#     function_parameters = extract_function_parameters(text_output)
#     print("Function Parameters:", function_parameters)
#     iter_flag = extract_iter(text_output)
#     print("ITER Flag:", iter_flag, type(iter_flag))


import json
import re

def extract_function_names(text):
    # Regex to capture JSON inside code blocks
    json_pattern = r'```(?:json)?\s*(.*?)\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)

    function_names = []
    for match in matches:
        try:
            # Parse JSON data
            json_data = json.loads(match)
            for item in json_data:
                function_name = item.get('function_name')
                if function_name:
                    function_names.append(function_name)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for function names: {e}")
    return function_names


def extract_function_parameters(text):
    # Regex to capture JSON inside code blocks
    json_pattern = r'```(?:json)?\s*(.*?)\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)

    parameter_list = []
    for match in matches:
        try:
            # Parse JSON data
            json_data = json.loads(match)
            for item in json_data:
                function_parameters = item.get('function_parameters')
                if function_parameters:
                    parameter_list.append(function_parameters)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for function parameters: {e}")
    return parameter_list


def extract_iter(text):
    # Regex to capture JSON inside code blocks
    json_pattern = r'```(?:json)?\s*(.*?)\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)

    iter_value = False  # Default value
    for match in matches:
        try:
            # Parse JSON data
            json_data = json.loads(match)
            for item in json_data:
                raw_iter = item.get("ITER")
                if raw_iter is not None:
                    # Convert ITER to a proper boolean
                    if isinstance(raw_iter, bool):
                        iter_value = raw_iter
                    elif isinstance(raw_iter, str):
                        iter_value = raw_iter.lower() == "true"
                    else:
                        print("ITER value is not boolean or string.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for ITER: {e}")
    return iter_value


if __name__ == "__main__":
    # Example usage:
    text_output = """
    ```json
[
    {
        "function_name": "coder",
        "function_parameters": {
            "message": "Certainly! Here is a simple Python code to add two numbers:\\n\\npython\\ndef add_two_numbers(a, b):\\n    return a + b\\n\\n# Example usage:\\nresult = add_two_numbers(3, 5)\\nprint('The sum is:', result)\\n"
        }
    }
]
    ```
    """

    function_names = extract_function_names(text_output)
    print("Function Names:", function_names)

    function_parameters = extract_function_parameters(text_output)
    print("Function Parameters:", function_parameters)

    iter_flag = extract_iter(text_output)
    print("ITER Flag:", iter_flag, type(iter_flag))
