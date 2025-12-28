from agent import *
import logging
from config import uneditable_files
import re
import llm
import os
import shutil
import time


def init_logger():
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    f_handler = logging.FileHandler('log.txt')
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

if __name__ == "__main__":
    delete_all_files_in_folder("logs")
    delete_all_files_in_folder("files")
    for file in uneditable_files:
        shutil.copy(file, f"files/{os.path.basename(file)}")



    git_commit("Initial commit")
    if os.path.exists('log.txt'):
        os.remove('log.txt')
    init_logger()

 
    agent_dict[config.ceo_name] = Agent(config.ceo_name, config.initial_prompt)
    begin_time = time.time()
    meta_output = get_llm_response([{"role": "system", "content": config.initial_prompt}], False)['choices'][0]['message']['content']
    print(meta_output)
    logging.info(meta_output)
    
    agent_pattern = re.compile(r'<agent name="(\w+)">(.*?)</agent>', re.DOTALL)
    agents = agent_pattern.findall(meta_output)

    # for agent in agents:
    #     if agent[0] == config.ceo_name:
    #         agent_dict[agent[0]] = Agent(agent[0], agent[1])
    
    for agent in agents:
        if agent[0] != config.ceo_name:
            agent_dict[config.ceo_name].add_subordinate(agent[0], "", agent[1])

    agent_dict[config.ceo_name].enqueue("user","Now let's start the project. Please split the task and talk to your subordinates to assign the tasks.")

    # Wait until all agents become idle
    last_log_time = time.time()
    while True:
        time.sleep(1)
        # Log status every 30 seconds to show the program is alive
        if time.time() - last_log_time > 30:
            running_agents = [name for name, agent in agent_dict.items() if agent.state == "running"]
            idle_agents = [name for name, agent in agent_dict.items() if agent.state == "idle"]
            logging.info(f"[HEARTBEAT] Running: {running_agents}, Idle: {idle_agents}")
            last_log_time = time.time()
        if all(agent.state == "idle" for agent in agent_dict.values()):
            ok = True
            for agent in agent_dict.values():
                try:
                    with git_lock:
                        f = open(f"files/todo_{agent.name}.txt", 'r')
                        content = f.read()
                        f.close()
                except FileNotFoundError:
                    content = ''
                if content != '':
                    agent.enqueue("system","Other agents have terminated. However, you still have unfinished tasks in your TODO list. Please finish them and clear it. If you are waiting for someone, chances are that they have forgotten about you. Please remind them.")
                    ok = False
            if not ok:
                continue
            else:
                logging.info(f"[MAIN] All agents idle. Sending final review to {config.ceo_name}")
                agent_dict[config.ceo_name].enqueue("system",r"All the agents have terminated. Please use read_file to browse and proofread all the output files. Be sure to test them if needed, and check whether the project has been completed(do not leave placeholders!). If the project is completed with 100% accuracy, please call the 'terminate' function; if not, please assign the remaining tasks.")
                ceo_wait_start = time.time()
                while(agent_dict[config.ceo_name].state != "idle"):
                    time.sleep(1)
                    if time.time() - ceo_wait_start > 60:
                        logging.info(f"[MAIN] Still waiting for {config.ceo_name} (state: {agent_dict[config.ceo_name].state})")
                        ceo_wait_start = time.time()
                if all(agent.state == "idle" for agent in agent_dict.values()):
                    break
                else:
                    continue
    end_time = time.time()
    total_tokens = input_token + output_token + llm.search_tokens
    logging.info(f"Time elapsed: {end_time-begin_time} seconds")
    logging.info(f"Input tokens: {input_token}, output tokens: {output_token}")
    logging.info(f"Search queries: {llm.search_queries}, search tokens: {llm.search_tokens}")
    logging.info(f"Number of agents: {len(agent_dict)}")
