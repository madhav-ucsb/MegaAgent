import chromadb
import time
import threading
import re
from llm import *
from utils import *
import config
import logging
from perplexity import perplexity_search, linkedin_profiles, fetch_profiles_text
chroma_client = chromadb.Client()

class Memory:
    def __init__(self, agent_name, initial_message):
        self.history = []
        self.history_pool = chroma_client.get_or_create_collection(name=agent_name)
        self.initial_message = initial_message
        self.name = agent_name
        self.subordinates = {}
        self.initialize_logger(agent_name)

    def add_memory(self, memory):
        if (memory['role'] not in ['function' , 'tool'] and memory['content'] != None):
            chroma_start = time.time()
            content_len = len(memory['content']) if memory['content'] else 0
            logging.info(f"CHROMA_ADD_START agent={self.name} content_length={content_len}")
            self.history_pool.add(documents=[memory['content']], ids=[str(time.time())])
            elapsed = time.time() - chroma_start
            logging.info(f"CHROMA_ADD_END agent={self.name} elapsed_seconds={elapsed:.3f}")
        self.logger.info(str(memory))
        self.history.append(memory)
    
    def add_subordinate(self, name, description, initial_prompt):
        self.subordinates[name] = description
        agent_dict[name] = Agent(name, initial_prompt+config.additional_prompt+"\nYour supervisor is: "+self.name)

    def get_subordinates(self):
        ret = ""
        for i in self.subordinates:
            ret+=f"{i}: {self.subordinates[i]}\n"
        return ret

    def add_dialogue(self, speaker, message):
        content = f"{speaker}: {message}"
        self.history.append({"role": "user", "content": content})
        self.logger.info(content)
        chroma_start = time.time()
        self.history_pool.add(documents=[content], ids=[str(time.time())])
        elapsed = time.time() - chroma_start
        logging.info(f"CHROMA_ADD agent={self.name} elapsed_seconds={elapsed:.3f}")
    
    def get(self):
        init = self.initial_message

        try:
            file_read_start = time.time()
            with git_lock:
                f = open(f"files/todo_{self.name}.txt", 'r')
                todo_list = f.read()
                f.close()
            logging.info(f"FILE_READ agent={self.name} file=todo elapsed_seconds={time.time() - file_read_start:.3f}")
        except FileNotFoundError:
            todo_list = 'You do not have a TODO list yet. You can use \'change_task_status\' function to create one.'
        if todo_list and todo_list != '':
            init+=f"\n\nHere is your current todo list: {todo_list}"

        try:
            file_read_start = time.time()
            with git_lock:
                f = open(f"files/status_{self.name}.txt", 'r')
                task_status = f.read()
                f.close()
            logging.info(f"FILE_READ agent={self.name} file=status elapsed_seconds={time.time() - file_read_start:.3f}")
        except FileNotFoundError:
            task_status = 'You do not have a task status yet. You can use \'change_task_status\' function to update it.'
        if task_status and task_status != '':
            init+=f"\n\nHere is your current task status(what you have done): {task_status}\nYou can use 'change_task_status' function to update it."
        else:
            init+=f"\n\nYou do not have a task status yet. You can use 'change_task_status' function to update it."

        if self.subordinates:
            init+="\n\nYou have added these subordinates. Do not add them again:\n"
            for i in self.subordinates:
                init+=f"{i}"
                if self.subordinates[i]!="":
                    init+=f": {self.subordinates[i]}"
                init+=",\n"
            init+="\nYou can talk to them by using the function called talk."

        from llm import input_token, output_token, search_tokens, search_queries
        total_tokens = input_token + output_token + search_tokens
        init+=f"\n\n=== Current Token Usage (All Agents) ===\n"
        init+=f"Input tokens: {input_token/1_000_000:.2f} Million\n"
        init+=f"Output tokens: {output_token/1_000_000:.2f} Million\n"
        init+=f"Search tokens: {search_tokens/1_000_000:.2f} Million\n"
        init+=f"Search queries: {search_queries:,}\n"
        init+=f"Total tokens: {total_tokens/1_000_000:.2f} Million"

        if self.history and self.history[-1]['content']:
            chroma_query_start = time.time()
            logging.info(f"CHROMA_QUERY_START agent={self.name}")
            relevant_history = self.history_pool.query(query_texts=self.history[-1]['content'], n_results=1)
            elapsed = time.time() - chroma_query_start
            logging.info(f"CHROMA_QUERY_END agent={self.name} elapsed_seconds={elapsed:.3f}")
            if relevant_history:
                init+=f"\n\nHere is a relevant memory: \n{relevant_history['documents'][0][0]}\nBelow is the recent dialogue."
        
        memory = [{"role": "system", "content": init}]
        if 'gpt-5' in config.model:
            memory[0]["role"] = "developer"
        for i in self.history[-config.MAX_MEMORY:]:
            memory.append(i)
        if len(memory) > 1 and ('gpt-5' in config.model or config.model == 'deepseek-chat' or config.model == 'deepseek-reasoner'):
            if memory[1]["role"] == "tool":
                if len(memory) == 1:
                    return memory[:1]
                else:
                    return [memory[0]] + memory[2:]
        

        return memory
    
    def initialize_logger(self, name):
        self.logger = logging.getLogger(name)

        # Create a file handler for logging
        file_handler = logging.FileHandler(f"logs/{name}.log", encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create a console handler for logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create a formatter that includes the time
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        # self.logger.addHandler(console_handler)
    
        # Set the logger level
        self.logger.setLevel(logging.INFO)

class Agent(Memory):
    def __init__(self, name, initial_message):
        super().__init__(name, initial_message)
        self.message_queue = []
        self.state = "idle"
        self.lock = threading.Lock()
        self.package = None
        # self.logger.info(f"Agent {name} initialized with initial message: {initial_message}")   
    
    def enqueue(self, speaker, message):
        with self.lock:
            self.message_queue.append({"role": speaker, "content": message})
        if self.state == "idle":
            threading.Thread(target=self.run, args=()).start()

    def execute(self, tool_name, tool_info, arguments, tool_call_id=None):
        tool_name = tool_name.lower()
        tool_info['name'] = tool_name
        try:
            if tool_name == 'exec_python_file':
                filename = arguments['filename']
                try:
                    result, self.package = start_interactive_subprocess(filename)
                except Exception as e:
                    result = f"Error: {e}"
                tool_info['content'] =f"Result of call to tool:"+ tool_name +":\n" + str(result)
                # self.logger.info(f"{filename}\n---Result---\n{result}")

            elif tool_name == 'input':
                content = arguments['content']
                if self.package:
                    try:
                        result, self.package = send_input(content,self.package)
                    except Exception as e:
                        result = f"Error: {e}"
                else:
                    result = "Error: No process to input."
                tool_info['content'] =f"Result of call to tool:"+ tool_name + "with arguments:\n" + str(arguments) +":\n" +   str(result)
                # self.logger.info(f"Input:\n{content}\n---Result---\n{result}")

            elif tool_name == 'read_file':
                tool_info['name'] = 'read_file'
                filename = arguments['filename']
                content, hashvalue = read_file(filename)
                result = f"{filename}\n---Content---\n{content}\n---base_commit_hash---\n{hashvalue}"
                tool_info['content'] =f"Result of call to tool:"+ tool_name +"with arguments:\n" + str(arguments) +":\n" + result
                # self.logger.info(result)
            
            elif tool_name == 'change_task_status':
                old_content, hashvalue = read_file(f"todo_{self.name}.txt")
                write_file(f"todo_{self.name}.txt", arguments['todo'], True, hashvalue, agent_name=self.name)

                old_content, hashvalue = read_file(f"status_{self.name}.txt")
                write_file(f"status_{self.name}.txt", arguments['done'], True, hashvalue, agent_name=self.name)

                tool_info['content'] = f"Success."
                # self.logger.info(f"Change todo list to: {arguments['todo']}\nChange task status to: {arguments['done']}")
            elif tool_name == 'perplexity_search':
                tool_info['name'] = 'perplexity_search'
                query = arguments['query']
                max_results = arguments['max_results']
                max_tokens_per_page = arguments['max_tokens_per_page']
                result = perplexity_search(query, max_results, max_tokens_per_page)
                tool_info['content'] =f"Result of call to tool:"+ tool_name +"with arguments:\n" + str(arguments) +":\n" + result
                self.logger.info(f"Query: {query}\nMax results: {max_results}\nMax tokens per page: {max_tokens_per_page}\nResult: {result}")
            elif tool_name == 'write_file':
                tool_info['name'] = 'write_file'
                filename = arguments['filename']
                content = arguments['content']
                if 'overwrite' in arguments:
                    overwrite = arguments['overwrite']
                    base_commit_hash = arguments['base_commit_hash'] if 'base_commit_hash' in arguments else None
                    result = write_file(filename, content, overwrite, base_commit_hash, agent_name=self.name, uneditable_files=config.uneditable_files)
                else:
                    result = write_file(filename, content, agent_name=self.name, uneditable_files=config.uneditable_files)
                tool_info['content'] =f"Result of call to tool:"+ tool_name +"with arguments:\n" + str(arguments) +":\n" + result
                self.logger.info(f"{filename}\n---Content---\n{content}\n---Result---\n{result}")
            elif tool_name == 'linkedin_profiles':
                tool_info['name'] = 'linkedin_profiles'
                query = arguments['query']
                start_page = 1
                if 'start_page' in arguments:
                    start_page = arguments['start_page']
   
                full_profile = False
                num_profiles = 10
                if 'full_profile' in arguments:
                    full_profile = arguments['full_profile']

                result = linkedin_profiles(query=query, start_page=start_page, full_profile=full_profile, num_profiles=num_profiles)
                tool_info['content'] =f"Result of call to tool:"+ tool_name +"with arguments:\n" + str(arguments) +":\n" + result
                self.logger.info(f"Query: {query}\nStart page: {start_page}\nFull profile: {full_profile}\nResult: {result}")
            elif tool_name == 'fetch_profiles_text':
                tool_info['name'] = 'fetch_profiles_text'
                urls = arguments['urls']
                result = fetch_profiles_text(urls)
                tool_info['content'] =f"Result of call to tool:"+ tool_name +"with arguments:\n" + str(arguments) +":\n" + result
                self.logger.info(f"URLs: {urls}\nResult: {result}")

            elif tool_name == 'add_agent':
                if len(self.subordinates) > config.MAX_SUBORDINATES:
                    tool_info['name'] = 'add_agent'
                    result = tool_info['content'] = f'Error: You have already recruited {config.MAX_SUBORDINATES} agents. No more agents can be recruited.\nRecruited agents: {self.get_subordinates()}'
                else:
                    tool_info['name'] = 'add_agent'
                    tool_info['content'] = "Result of call to tool:"+ tool_name +"with arguments:\n" + str(arguments) +":\n" + 'Success.'
                    agent_name = arguments['name']
                    if agent_name in used_names:
                        # tool_info['content'] = f"Error: {agent_name} already exists. Please use another name."
                        # logging.error(f"Error: {agent_name} already exists. Please use another name.")
                        agent_name = agent_name + '_' + str(time.time())[-5:]
                        tool_info['content'] = f"Warning: {arguments['name']} already exists. Automatically change to {agent_name}.\nSuccess."
                    self.add_subordinate(agent_name,arguments['description'], arguments['initial_prompt'])
                    self.logger.info(f"Add agent: {agent_name} success, with description: {arguments['description']}, and initial_prompt: {arguments['initial_prompt']}")
            elif tool_name == 'talk':
                pattern = re.compile(r'<talk goal=(?:"([^"]+)"|\'([^\']+)\')>(.*?)</talk>', re.IGNORECASE | re.DOTALL)
                matches = re.findall(pattern, arguments['messages'])
                tool_info['name'] = 'talk'
                result = arguments['messages']
                if matches:
                    tool_info['content'] = 'Successfully sent. You may terminate now to wait for the response, or complete the rest of your TODO list first.'
                else:
                    tool_info['content'] = 'Error: No matched <talk goal="Name"></talk> found in your response. You must talk in this specific XML format.'
                tool_info['content'] = "Result of call to tool:"+ tool_name +"with arguments:\n" + str(arguments) +":\n" + tool_info['content']
                for match in matches:
                    name = match[0] if match[0] else match[1]
                    content = match[2].strip()
                    
                    if name in agent_dict:
                        agent_dict[name].enqueue("user", f"{self.name} : {content}")
                    else:
                        raise ValueError(f"Error: {name} is not a valid agent name")
                     
            elif tool_name == 'terminate':
                try:
                    with git_lock:
                        f = open(f"files/todo_{self.name}.txt", 'r')
                        content = f.read()
                        f.close()
                except FileNotFoundError:
                    content = ''
                # if content != None and content != '':
                #     tool_info['content'] = f"Error: You have not finished your TODO list. That is:\n\n{content}\n\n Please finish it and clear it by calling 'change_task_status' function."
                # else:
                return {}
            else:
                raise ValueError(f"Error: {tool_name} is not a valid function name")
            return tool_info
        except Exception as e:
            self.logger.error(e)
            return {"role": "user", "content": "Error:"+str(e)}

    def run(self):
        if(self.state!="idle"):
            return
        self.state = "running"
        try:
            while self.message_queue:
                previous_memory = self.get()
                
                with self.lock:
                    new_message = self.message_queue
                    self.message_queue = []
                
                for i in new_message:
                    logging.info(f"New message: {i}")
                    self.add_memory(i)
                    
                req = previous_memory + new_message
                round = 0
                self.package = None
                assistant_output = None
                function_call_check = 'tool_calls' if 'gpt-5' in config.model or config.model == 'deepseek-chat' or config.model == 'deepseek-reasoner' else 'function_call'

                while round < config.MAX_ROUNDS:
                    try:
                        response = get_llm_response(req, agent_name=self.name)
                        assistant_output = response['choices'][0]['message']
                        llm_output = assistant_output['content']
                    except Exception as e:
                        error_data = {
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'agent_name': self.name,
                            'error': str(e),
                            'request': req
                        }
                        try:
                            with open('error_response.json', 'w', encoding='utf-8') as f:
                                json.dump(error_data, f, indent=2, ensure_ascii=False)
                            self.logger.error(f"LLM call failed, logged to error_response.json: {e}")
                        except Exception as write_error:
                            self.logger.error(f"Failed to write error_response.json: {write_error}")
                        
                        try:
                            req_json = json.dumps(req, ensure_ascii=False)
                            with open('log.txt', 'a', encoding='utf-8') as f:
                                f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}: ERROR_REQUEST_DUMP: {req_json}\n")
                                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: EXCEPTION: {str(e)}\n")
                                f.flush()
                        except:
                            pass
                        raise

                    self.add_memory(assistant_output)
                    req = self.get()
                    if llm_output != None:
                        self.logger.info(f"Assistant: {json.dumps(assistant_output, indent=4)}")
                    if function_call_check not in assistant_output:
                        self.add_dialogue("user", "Error: No function call found in the response. You must use function calls to work and communicate with other agents. If you have nothing to do now, please call 'terminate' function.")
                        req = self.get()
                        round += 1
                    else:
                        break

                round = 0
                while round < config.MAX_ROUNDS:
                    if function_call_check == 'function_call':
                        print("assistant_output", assistant_output)
                        tool_call = assistant_output[function_call_check]
                        tool_name = tool_call['name']
                        arguments = json.loads(tool_call['arguments'])
                        tool_info = self.execute(tool_name, {"role": "function"}, arguments)
                        if tool_info == {}:
                            break
                        self.add_memory(tool_info)
                        req += [tool_info]
                        round += 1
                        response = get_llm_response(req, agent_name=self.name)
                        assistant_output = response['choices'][0]['message']
                        llm_output = assistant_output['content']
                        self.add_memory(assistant_output)
                    elif function_call_check == 'tool_calls':
                        logging.info(f"Assistant output: {assistant_output}")
                        tool_responses = []
                        
                        for tool_call in assistant_output['tool_calls']:
                            logging.info(f"Tool call: {tool_call}")
                            tool_name = tool_call['function']['name']
                            arguments = json.loads(tool_call['function']['arguments'])
                            tool_call_id = tool_call['id']
                            logging.info(f"Tool call: {tool_name}, arguments: {arguments}")
                            tool_result = self.execute(tool_name, {"role": "assistant"}, arguments, tool_call_id)
                            logging.info(f"Tool result: {tool_result}")
                            try:
                                req_json_pretty = json.dumps(req, indent=2, ensure_ascii=False)
                                req_json_compact = json.dumps(req, ensure_ascii=False)
                                print("Request:" , req_json_pretty)
                                
                                logging.info(f"Request length: {len(req_json_compact)} characters")
                                
                                if len(req_json_compact) > 10000:
                                    logging.info("Request (large, logging in chunks):")
                                    chunk_size = 5000
                                    for i in range(0, len(req_json_compact), chunk_size):
                                        chunk = req_json_compact[i:i+chunk_size]
                                        logging.info(f"Request chunk {i//chunk_size + 1}: {chunk}")
                                else:
                                    logging.info(f"Request: {req_json_compact}")
                                
                                for handler in logging.getLogger().handlers:
                                    handler.flush()
                                
                                with open('log.txt', 'a', encoding='utf-8') as f:
                                    f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}: REQUEST_DUMP: {req_json_compact}\n")
                                    f.flush()
                                
                                logging.info("Request logged successfully")
                            except Exception as e:
                                logging.error(f"Failed to log request: {e}")
                                try:
                                    with open('log.txt', 'a', encoding='utf-8') as f:
                                        f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}: REQUEST_ERROR but here's repr: {repr(req)}\n")
                                except:
                                    pass
                            tool_response = {"role": "tool", "tool_call_id": tool_call_id, "content": ""}
                            if tool_result != {}:
                                tool_response = {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": tool_result['content'] 
                                }
                            
                            tool_responses.append(tool_response)
                            self.add_memory(tool_response)
                            if tool_result == {}:
                                break
                        
                        if tool_result == {}:
                            break
                        req += tool_responses
                        round += len(tool_responses)
                        response = get_llm_response(req, agent_name=self.name)
                        assistant_output = response['choices'][0]['message']
                        llm_output = assistant_output['content']
                        self.add_memory(assistant_output)
                    req += [assistant_output]
                    while function_call_check not in assistant_output:
                        req += [{"role":"user", "content": "Error: No function call found in the response. You must use function calls to work and communicate with other agents. If you have nothing to do now, please call 'terminate' function."}]
                        try:
                            response = get_llm_response(req, agent_name=self.name)
                            assistant_output = response['choices'][0]['message']
                            llm_output = assistant_output['content']
                            self.add_memory(assistant_output)
                            req += [assistant_output]
                            round += 1
                        except Exception as e:
                            error_data = {
                                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                                'agent_name': self.name,
                                'error': str(e),
                                'request': req
                            }
                            try:
                                with open('error_response.json', 'w', encoding='utf-8') as f:
                                    json.dump(error_data, f, indent=2, ensure_ascii=False)
                                self.logger.error(f"LLM call failed in error recovery loop, logged to error_response.json: {e}")
                            except Exception as write_error:
                                self.logger.error(f"Failed to write error_response.json: {write_error}")
                            
                            try:
                                req_json = json.dumps(req, ensure_ascii=False)
                                with open('log.txt', 'a', encoding='utf-8') as f:
                                    f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}: ERROR_REQUEST_DUMP: {req_json}\n")
                                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: EXCEPTION: {str(e)}\n")
                                    f.flush()
                            except:
                                pass
                            raise

                    if assistant_output and function_call_check in assistant_output and function_call_check == 'tool_calls':
                            for tool_call in assistant_output['tool_calls']:
                                tool_name = tool_call['function']['name']
                                arguments = json.loads(tool_call['function']['arguments'])
                                tool_call_id = tool_call['id']
                                tool_result = self.execute(tool_name, {"role": "assistant"}, arguments, tool_call_id)
                                tool_response = {"role": "tool", "tool_call_id": tool_call_id, "content": ""}
                                if tool_result != {}:
                                    tool_response["content"] = tool_result['content']
                                self.add_memory(tool_response)
        except Exception as e:
            logging.exception(f"Agent.run uncaught exception name={self.name}")
            try:
                self.logger.exception(f"Agent.run uncaught exception name={self.name}")
            except Exception:
                pass
        finally:
            self.state = "idle"

agent_dict = {}