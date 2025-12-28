import config
import requests
import os
import json
import logging
from perplexity import perplexity_search
import time
written_files = dict()
used_names = set()
tools = []
input_token=0
output_token=0
search_queries=0
search_tokens=0

DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_READ_TIMEOUT_SECONDS = 60

def gen_tools(agent_name):
    if config.share_file:
        wf = []
        for agent, files in written_files.items():
            agent_files = [f for f in files if 'todo' not in f.lower() and 'status' not in f.lower()]
            wf.extend(agent_files)
    else:
        if agent_name in written_files:
            wf = [f for f in written_files[agent_name] if 'todo' not in f.lower() and 'status' not in f.lower()]
        else:
            wf = []
    global tools
    tools = [
        {
                "name": "exec_python_file",
                "description": "Execute a Python file and get the result. Cannot detect bugs. Be sure to review the code or use write_file to write the file first. If the program requires user input, please use this function first, and then use 'input' function to pass your input.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "The filename of the Python file to be executed."
                        }
                    }
            }
        },
        {
                "name": "input",
                "description": "Input a string to the running Python code. Only available after exec_python_file is called.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The string to be input."
                        }
                    }
                }
        },
        {
                "name": "read_file",
                "description": f"Read the content of a file. Return file content and file hash. To modify a file, please first read it, then write it(using the same hash).\nYou have created these files:{wf}\nYou can also read any files created by other agents.",
                "parameters": {
                            "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "The filename to be read."
                        }
                    }
                },
                "required": [
                    "filename"
                ]
        },
        {
                "name": "write_file",
                "description": f"Write raw content to a file. If the file exists, only overwrite when overwrite = True and hash value (get it from read_file) is correct. ",
                "parameters": {
                            "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "The filename to be written."
                        },
                        "content": {
                            "type": "string",
                            "description": r"The content to be written. Use \n instead of \\n for a new line."
                        },
                        "overwrite": {
                            "type": "boolean",
                            "description": "Optional. Whether to overwrite the file if it exists. Default is False. If True, base_commit_hash is required."
                        },
                        "base_commit_hash": {
                            "type": "string",
                            "description": "Optional. The hash value of the file to be modified(get it from read_file). Required when overwrite = True."
                        }
                    }
                },
                "required": [
                    "filename",
                    "content"
                ]
        },
        {"name": "change_task_status",
                "description": "Change the content of your task status. It should include what you have done, and what you should do later, in case you would forget.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "todo": {
                            "type": "string",
                            "description": "You TODO list. You must finish all the tasks and clear this list before calling the 'terminate' function."
                        },
                        "done": {
                            "type": "string",
                            "description": "The tasks you have done. You can write anything you want to remember."
                        }
                    }
                },
                "required": [
                    "content"
                ]
        },
        {
                "name": "add_agent",
                "description": "If your task is too complex or if your reasoning indicates it make more sense to distribute the tasks atomically, you can recruit agents to the conversation as your subordinates and help you. Return the real name. To add multiple agents, please call this function multiple times. After that, you MUST talk to them using function calls. You do not need to add your collaborators in the prompt, since they have been added by the CEO.",
                "parameters": {
                            "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the agent to be added. Do not use space. Do not use the same name as any existing agent."
                        },
                        "description": {
                            "type": "string",
                            "description": "The description of the agent, for your reference."
                        },
                        "initial_prompt": {
                            "type": "string",
                            "description": '''
                            The initial prompt and memory of that agent. Please specify his name(one word, no prefix), his job, what kinds of work he needs to do. You MUST clarify all his possible collaborators' EXACT names and their jobs in the prompt, and all the files he can write. The format should be like (The example is for Alice in another novel writing project):
                            You are Alice, a novelist. Your job is to write a single chapter of a novel with 1000 words according to the outline (outline.txt) from Carol, the architect designer, and pass it to David (chapter_x.txt), the editor. Please only follow this routine. Your collarborators include Bob(the Boss), Carol(the architect designer) and David(the editor).
                            Please note that every agent is lazy, and will not care anything not mentioned by your prompt. To ensure the completion of the project, the work of each agent should be non-divisable, detailed in specific action(like what file to write) and limited to a simple and specific instruction(For instance, instead of "align with the overall national policies", please specify those policies).
                            '''
                        }
                    }
                },
                "required": [
                    "name",
                    "description",
                    "initial_prompt"
                ]
        }
    ]
    
    # Conditionally add LinkedIn tools if enabled
    if config.linkedin_enabled:
        tools.extend([
            {
                "name": "fetch_profiles_text",
                "description": "Fetch the text content of LinkedIn profiles based on a list of URLs. The cost is $0.001 per url.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of URLs to fetch text content from."
                        }
                    },
                    "required": [
                        "urls"
                    ]
                }
            },
            {
                "name": "linkedin_profiles",
                "description": "Search for LinkedIn profiles based on a query that will enter into google search confined to the url for linkedin profiles. and optional parameters.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query string. Be specific and clear about what you're looking for."
                        },
                        "start_page": {
                            "type": "integer",
                            "description": "The starting page for the search results. Default is 1. Never overlap. If you need to split a query, keep track of the starting_page. Each page has 10 profiles. If you collect 200 profiles over 2 queries, first start_page 100 results then start_page 10 100 results. Use common sense to determine the number of profiles needed."
                        },
                        "num_profiles": {
                            "type": "integer",
                            "description": "Optional. The number of profiles to retrieve. It costs $0.03 per 1k profiles. 1$ per 1k profiles with full_profile=True."
                        },
                        "full_profile": {
                            "type": "boolean",
                            "description": "The text to retrieve full profile information. Default is False. If you want to fetch full profile information, keep in mind to lower the num_profiles or split into multiple steps if you are operating a large important query because on each profile has hundreds of tokens. The number of profiles is limited to 10 or in other words, one page of search results if full_profile=True."
                        }
                    },
                    "required": [
                        "query",
                        "num_profiles",
                    ]
                }
            }
        ])
    
    # Continue with remaining tools
    tools.extend([
        {
                "name": "talk",
                "description": "Leave a message to specific agents for feedback. They will reply you later on.",
                "parameters":{
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "string",
                            "description": "All the messages to be sent. The format must look like: <talk goal=\"Name\">TalkContent</talk><talk goal=\"Name\">TalkContent</talk>"
                        }
                    }
                },
                "required": [
                    "messages"
                ]
        },
        {
                "name": "terminate",
                "description": "End your current conversation. Please ensure all your tasks in your TODO list have been done and cleared.",
        },
        {
                "name": "perplexity_search",
                "description": "Perform a web search using the Perplexity API to find information on the internet. This tool is useful for finding current information, research, facts, and general knowledge. The search returns ranked results with titles, URLs, and content snippets that are ai generated. Keep in mind, if you need multiple pieces of information, you should use multiple queries if one query would not be sufficient. Keep in mind it only returns text. There is no desire to download files from urls or extract better content. This api does the best job of extracting content. If you think more content is needed, complete a different search or increase the max_tokens.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query string. Be specific and clear about what you're looking for."
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of search results to return. Must be between 1 and 20. Default is 5. Use lower values for quick searches, higher values for comprehensive research.",
                            "default": 5
                        },
                        "max_tokens_per_page": {
                            "type": "integer",
                            "description": "Maximum number of tokens to retrieve from each webpage. Must be between 256 and 2048. Default is 512. Higher values (1024-2048) provide more comprehensive content but increase processing time and latency. Lower values (256-512) are faster but may have less detailed results.",
                            "default": 512
                        }
                    },
                    "required": ["query"]
                }
        }
    ])


def _get_llm_response(messages, enable_tools=True, agent_name=''):
    api_key = config.api_key
    url = config.url
    headers = {'Content-Type': 'application/json',
            'Authorization':f'Bearer {api_key}'}
    gen_tools(agent_name)
    req_json = json.dumps(messages, ensure_ascii=False)
    with open('log.txt', 'a', encoding='utf-8') as f:
        f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}: LLM_REQUEST_DUMP: {req_json}\n")
        f.flush()
    if "tool_calls" in messages[-1]:
        messages = messages[:-1]
    if len(messages) > 1:

        if "tool_calls" in messages[-2] and messages[-1]["role"] != "tool":
            messages = messages[:-2] + messages[-1:]
    # Remove leading tool messages that come before the first tool_calls
    if config.model in ["gpt-5", 'gpt-5-mini', "deepseek-chat", "deepseek-reasoner"]:
        first_tool_calls_index = len(messages)
        for i, msg in enumerate(messages):
            if "tool_calls" in msg:
                first_tool_calls_index = i
                break
        filtered_messages = []
        for i, msg in enumerate(messages):
            if i >= first_tool_calls_index or msg["role"] != "tool":
                filtered_messages.append(msg)
        messages = filtered_messages
    # Remove duplicate messages while preserving order
    seen = set()
    unique_messages = []
    for msg in messages:
        if msg["role"] != "tool" or config.model not in ['gpt-5', 'gpt-5-mini', 'deepseek-chat', 'deepseek-reasoner']:
            unique_messages.append(msg)
        else:
            tool_call_id = msg["tool_call_id"]
            if tool_call_id not in seen:
                seen.add(tool_call_id)
                unique_messages.append(msg)
    messages = unique_messages
    
    
    if enable_tools:
        body = {
            'model': config.model,
            "messages": messages,
            "functions": tools,
            "temperature": 0,
        }
        if 'gpt-5' in config.model:
            body = {
                'model': config.model,
                "messages": messages,
                "tools": [{"type": "function", "function": tool} for tool in tools],
                "reasoning_effort" : "minimal",
                "verbosity": "low",
                
            }
            if 'responses' in config.url:
                body['reasoning']['effort'] = 'low'
                body['text']['verbosity'] = 'low'
        if config.model == 'deepseek-reasoner' or config.model == 'deepseek-chat':
            body = {
                'model': config.model,
                "messages": messages,
                "tools": [{"type": "function", "function": tool} for tool in tools],
                
            }
    else:
        body = {
            'model': config.model,
            "messages": messages,
            "temperature": 0,
        }
        if 'gpt-5' in config.model:
            body = {
                'model': config.model,
                "messages": messages,
                "reasoning_effort": "low",
                "verbosity": "low"
            }
        
    try:
        request_started = time.time()
        logging.info(f"LLM_HTTP_START agent={agent_name} model={config.model}")
        response = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=(DEFAULT_CONNECT_TIMEOUT_SECONDS, DEFAULT_READ_TIMEOUT_SECONDS),
        )
        elapsed = time.time() - request_started
        logging.info(f"LLM_HTTP_END agent={agent_name} model={config.model} status={getattr(response, 'status_code', None)} elapsed_seconds={elapsed:.3f}")
        try:
            return response.json()
        except Exception as e:
            return {
                'error': f"Failed to parse JSON response: {e}",
                'status_code': getattr(response, 'status_code', None),
                'text': getattr(response, 'text', '')[:2000],
            }
    except Exception as e:
        return {'error': e}

def get_llm_response(messages, enable_tools=True, agent_name=''):
    response = _get_llm_response(messages, enable_tools, agent_name)
    try:
        logging.info(f"LLM_RESPONSE agent={agent_name} keys={list(response.keys())}")
    except Exception:
        logging.info(f"LLM_RESPONSE agent={agent_name} type={type(response)}")
    
    # Check if response contains an error
    if 'error' in response:
        error_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'agent_name': agent_name,
            'error': response['error'],
            'request': messages
        }
        
        # Log to error_response.json
        try:
            with open('error_response.json', 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
            logging.error(f"LLM Error logged to error_response.json: {response['error']}")
        except Exception as e:
            logging.error(f"Failed to write error_response.json: {e}")
        
        # Also log in REQUEST_DUMP format to log.txt
        try:
            req_json = json.dumps(messages, ensure_ascii=False)
            with open('log.txt', 'a', encoding='utf-8') as f:
                f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}: ERROR_REQUEST_DUMP: {req_json}\n")
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: ERROR_DETAILS: {json.dumps(response['error'], ensure_ascii=False)}\n")
                f.flush()
        except Exception as e:
            logging.error(f"Failed to write error dump to log.txt: {e}")
    
    retries = 0
    while 'choices' not in response:
        logging.error(response)
        time.sleep(1)
        retries += 1
        if retries >= 20:
            raise Exception(f"LLM response missing 'choices' after {retries} retries")
        response = _get_llm_response(messages, enable_tools, agent_name)
    # if response['choices'][0]['message']['content']:
    #     logging.info(response['choices'][0]['message']['content'])
    global input_token,output_token
    input_token+=response['usage']['prompt_tokens']
    output_token+=response['usage']['completion_tokens']
    logging.info(f"Input token: {input_token}, Output token: {output_token}")
    return response
