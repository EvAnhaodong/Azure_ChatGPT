# Azure_ChatGPT

According to the [Azure OpenAI Service Documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/chatgpt-quickstart?tabs=command-line&pivots=programming-language-python), the GPT-35-Turbo model for Azure OpenAI service only has completion functionality, and cannot perform conversations like the official OpenAI API. 

**[2023/3/23 update] Azure update api_version: 2023-03-15-preview. Now we can use openai.ChatCompletion for gpt-3.5-turbo and gpt4.**

Based on the chat prompt construction method provided in the documentation.

I have made modifications to the source code of the V3 module in [ChatGPT](https://github.com/acheong08/ChatGPT).

Encapsulate Azure OpenAI GPT-3.5-Turbo model and GPT-4 model into a chat API. Extensible for chatbots etc.

# Installation

`python3 -m pip install azureChatGPT --upgrade`

# Terminal Chat

Get `api_key`, `api_base`, `engine_gpt-3.5-turbo`, `engine_gpt-4` from your Azure (API_KEY, ENDPOINT, ENGINE)

`engine` set default engine "gpt-3.5-turbo" or "gpt-4"

Modify the configuration file `azure.yaml` and run 

## Command line

`python3 -m azureChatGPT --config azure.yaml`

> You can save or load your YAML at any time during the program's execution, making it easy to restore your work state (including conversation records).

```
 $ python3 -m azureChatGPT -h

    ChatGPT - Official azureChatGPT API
    Repo: github.com/EvAnhaodong/Azure_ChatGPT
    
Type '!help' to show a full list of commands
Press Esc followed by Enter or Alt+Enter to send a message.

usage: __main__.py [-h] [--temperature TEMPERATURE] [--top_p TOP_P] [--system_prompt SYSTEM_PROMPT] --config CONFIG [--submit_key SUBMIT_KEY]

options:
  -h, --help            show this help message and exit
  --temperature TEMPERATURE
                        Temperature for response
  --top_p TOP_P         Top p for response
  --system_prompt SYSTEM_PROMPT
                        Base prompt for chatbot
  --config CONFIG       Path to config yaml file
  --submit_key SUBMIT_KEY
                        Custom submit key for chatbot. For more information on keys, see https://python-prompt-toolkit.readthedocs.io/en/stable/pages/advanced_topics/key_bindings.html#list-of-special-keys
```

# Disclaimers

This is a personal project. Modify from [ChatGPT](https://github.com/acheong08/ChatGPT)
