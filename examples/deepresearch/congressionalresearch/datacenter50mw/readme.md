

## How to Reproduce
1. Import openai_key and perplexity_key in config.py
2. Ask gpt-5 or any llm to come up with a prompts for config.py
3. In `examples/congressional/`, run:
   ```bash
   python main.py
   ```
4. The run will:
   - Initialize agents and logs in `logs/`
   - Write intermediate and final results to `files/`
   - Produce `files/ans.txt` upon aggregation

Requirements and LLM/tool configuration are defined in `examples/congressional/config.py`.

## View examples
- Look at the files folder in each of these 3 examples
