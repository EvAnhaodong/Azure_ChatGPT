"""
A simple wrapper for the official azure ChatGPT API
"""
import argparse
import yaml
import os
import sys
from typing import NoReturn
import time
from itertools import cycle

import base64
from mimetypes import guess_type

from openai import AzureOpenAI
import boto3
import json
import tiktoken

from .utils import create_completer
from .utils import create_keybindings
from .utils import create_session
from .utils import get_filtered_keys_from_object
from .utils import get_input


class Chatbot:
    """
    Official ChatGPT API
    """

    def __init__(
        self,
        api_key: str = "",
        engine: str = "",
        api_base: str = "",
        api_version: str = "2024-02-01",
        max_tokens: dict = {"gpt-4-turbo": 6000, "gpt-4":4000,"gpt-4o":6000,"claude3_haiku":6000, "claude3_sonnet":6000},
        temperature: float = 0.5,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally",
    ) -> None:

        self.engine = engine
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.api_version = api_version
        self.api_key = api_key
        self.api_base = api_base

        self.init_openai()

        self.conversation: dict = {
            "default": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ],
        }

    def init_openai(self):
        self.api = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=f"{self.api_base}"
        )
    

    def init_claude(self):
        self.api = boto3.client("bedrock-runtime", region_name="us-east-1")


    def add_to_conversation(
        self,
        message: str,
        role: str,
        name: str = "",
        convo_id: str = "default",
        image: str = "",
    ) -> None:
        """
        Add a message to the conversation
        """
        if image:
            image_format = guess_type(image)[0]
            image_base64 = base64.b64encode(open(image, 'rb').read()).decode('utf-8')
            if 'claude3' in self.engine:
                content=[{"type": "text", "text": message}, 
                        {"type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"{image_format}",
                                "data": f"{image_base64}",
                    }}]
            else:
                content=[{"type": "text", "text": message}, {"type": "image_url", "image_url": {"url": f"data:{image_format};base64,{image_base64}"}}]
        else:
            content=[{"type": "text", "text": message}]

        if not name or 'claude3' in self.engine:
            self.conversation[convo_id].append({"role": role, "content": content})
        else:
            self.conversation[convo_id].append(
                {"role": role, "name": name, "content": content}
            )

    def __truncate_conversation(self, convo_id: str = "default") -> None:
        """
        Truncate the conversation
        """
        self.conversation["current"] = [
            dict((key, sentence[key]) for key in ["role", "content"])
            for sentence in self.conversation[convo_id]
        ]
        while True:
            if (
                self.get_token_count("current", skip_system = True) > self.max_tokens[self.engine]
                and len(self.conversation["current"]) > 1
            ):
                # Don't remove the first message
                del self.conversation["current"][1]
            else:
                break

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def get_token_count(self, convo_id: str = "default", skip_system = False) -> int:
        """
        Get token count
        """
        encoding = tiktoken.encoding_for_model("gpt-4o")
        num_tokens = 0
        for message in self.conversation[convo_id]:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            if skip_system and message['role'] == "system":
                continue
            num_tokens += 3
            for key, value in message.items():
                if isinstance(value,list):
                    for content in value:
                        num_tokens += len(encoding.encode(content.get("text", "")))
                else:
                    num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += 1
        num_tokens += 3  # every reply is primed with <im_start>assistant
        return num_tokens

    def get_max_tokens(self, convo_id: str = "default") -> int:
        """
        Get remaining tokens
        """
        return 2000

    def switch_engine(self):
        self.engine = next(self.engine_list)
        self.__dict__.update(getattr(self, "engine_" + self.engine))
        if 'claude3' in self.engine:
            self.init_claude()
        else:
            self.init_openai()

    def ask_stream(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        image:str = "",
        **kwargs,
    ) -> str:
        """
        Ask a question
        """
        # Make conversation if it doesn't exist
        if convo_id not in self.conversation:
            self.reset(convo_id=convo_id, system_prompt=self.system_prompt)
        self.add_to_conversation(prompt, role, convo_id=convo_id, image=image)
        self.__truncate_conversation(convo_id=convo_id)
        response_role: str = "assistant"
        full_response: str = ""
        model: str = ""
        if 'claude3' in self.engine:
            body = json.dumps({
                        "system": self.conversation["current"][0]['content'],
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": self.get_max_tokens(),
                        "top_p": kwargs.get("top_p", self.top_p),
                        "temperature": kwargs.get("temperature", self.temperature),
                        "messages": self.conversation["current"][1:]
                    }, ensure_ascii=False)
            response = self.api.invoke_model_with_response_stream(
                            body=body, modelId=self.model)
            for resp in response.get('body'):
                time.sleep(0.01)
                model = self.model
                chunk = resp.get('chunk')
                if chunk:
                    if 'delta' in json.loads(chunk.get('bytes').decode()).keys():
                            content = json.loads(chunk.get('bytes').decode()).get('delta').get('text')
                            if content:
                                full_response+=content
                                yield content

        else:
            response = self.api.chat.completions.create(
                messages=self.conversation["current"],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=self.get_max_tokens(),
                top_p=kwargs.get("top_p", self.top_p),
                frequency_penalty=kwargs.get(
                    "frequency_penalty",
                    self.frequency_penalty,
                ),
                presence_penalty=kwargs.get(
                    "presence_penalty",
                    self.presence_penalty,
                ),
                model=self.model,
                stream=True,
            )
            for resp in response:
                time.sleep(0.01)
                model = resp.model
                choices = resp.choices
                if not choices:
                    continue
                delta = choices[0].delta
                if not delta:
                    continue
                content = delta.content
                if not content:
                    continue
                full_response += content
                yield content

        self.add_to_conversation(full_response, response_role, model, convo_id=convo_id)
        # print(self.conversation['current'])

    def ask(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        image:str = "",
        **kwargs,
    ) -> str:
        response = self.ask_stream(
            prompt=prompt,
            role=role,
            convo_id=convo_id,
            image=image,
            **kwargs,
        )
        full_response: str = "".join(response)
        return full_response

    def rollback(self, n: int = 1, convo_id: str = "default") -> None:
        """
        Rollback the conversation
        """
        for _ in range(n):
            self.conversation[convo_id].pop()

    def reset(self, convo_id: str = "default", system_prompt: str = None) -> None:
        """
        Reset the conversation
        """
        self.conversation[convo_id] = [
            {"role": "system", "content": system_prompt or self.system_prompt},
        ]

        if self.get_token_count(convo_id) > min(self.max_tokens.values()):
            raise Exception("System prompt is too long")

    def save(self, file: str, *keys: str) -> None:
        """
        Save the Chatbot configuration to a JSON file
        """
        with open(file, "w", encoding="utf-8") as f:
            yaml.dump(
                {
                    key: self.__dict__[key]
                    for key in get_filtered_keys_from_object(self, *keys)
                },
                f,
            )

    def load(self, file: str, *keys: str) -> None:
        """
        Load the Chatbot configuration from a JSON file
        """
        with open(file, encoding="utf-8") as f:
            loaded_config = yaml.load(f, Loader=yaml.FullLoader)
            self.__dict__.update(loaded_config)
            self.engine_list = cycle(self.engine_list)
            self.engine = next(self.engine_list)
            self.__dict__.update(getattr(self, "engine_" + self.engine))
            if 'claude3' in self.engine:
                self.init_claude()
            else:
                self.init_openai()


class ChatbotCLI(Chatbot):
    def print_config(self, convo_id: str = "default") -> None:
        """
        Prints the current configuration
        """
        print(
            f"""
ChatGPT Configuration:
  Conversation ID:  {convo_id}
  Messages:         {len(self.conversation[convo_id])}
  Tokens used:      {( num_tokens := self.get_token_count(convo_id) )} / {self.max_tokens}
  Cost:             {"${:.5f}".format(( num_tokens / 1000 ) * 0.002)}
  Engine:           {self.engine}
  Temperature:      {self.temperature}
  Top_p:            {self.top_p}
            """,
        )

    def print_help(self) -> None:
        """
        Prints the help message
        """
        print(
            """
Commands:
  !help             Display this message
  !switch           Switch engine
  !rollback n       Rollback the conversation by n messages
  !save file [keys] Save the Chatbot configuration to a JSON file
  !load file [keys] Load the Chatbot configuration from a JSON file
  !reset            Reset the conversation
  !exit             Quit chat

Config Commands:
  !config           Display the current config
  !temperature n    Set the temperature to n
  !top_p n          Set the top_p to n
  !openai [keys]    Display the openai config

Examples:
  !save c.json               Saves all ChatbotGPT class variables to c.yaml
  !save c.json engine top_p  Saves only temperature and top_p to c.yaml
  !load c.json not engine    Loads all but engine from c.yaml
  !load c.json session       Loads session proxies from c.yaml


  """,
        )

    def handle_commands(self, input: str, convo_id: str = "default") -> bool:
        """
        Handle chatbot commands
        """
        command, *value = input.strip().split(" ")
        if command == "!help":
            self.print_help()
        elif command == "!exit":
            sys.exit()
        elif command == "!reset":
            self.reset(convo_id=convo_id)
            print("\nConversation has been reset")
        elif command == "!switch":
            self.switch_engine()
        elif command == "!config":
            self.print_config(convo_id=convo_id)
        elif command == "!rollback":
            self.rollback(int(value[0]), convo_id=convo_id)
            print(f"\nRolled back by {value[0]} messages")
        elif command == "!save":
            self.save(*value)
            print(
                f"Saved {', '.join(value[1:]) if len(value) > 1 else 'all'} keys to {value[0]}",
            )
        elif command == "!load":
            self.load(*value)
            print(
                f"Loaded {', '.join(value[1:]) if len(value) > 1 else 'all'} keys from {value[0]}",
            )
        elif command == "!temperature":
            self.temperature = float(value[0])
            print(f"\nTemperature set to {value[0]}")
        elif command == "!top_p":
            self.top_p = float(value[0])
            print(f"\nTop_p set to {value[0]}")
        elif command == "!openai":
            if len(value) > 0:
                for i in value:
                    print(f"{i}    {getattr(self,i)}")
            else:
                for i in ["api_type", "api_key", "api_base"]:
                    print(f"{i}    {getattr(self,i)}")
        else:
            raise Exception("Wrong command")

        return True


def main() -> NoReturn:
    """
    Main function
    """
    print(
        """
    ChatGPT - Official Azure_ChatGPT API
    Repo: github.com/EvAnhaodong/Azure_ChatGPT
    """,
    )
    print("Type '!help' to show a full list of commands")
    print("Press Esc followed by Enter or Alt+Enter to send a message.\n")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for response",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1,
        help="Top p for response",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are ChatGPT, a large language model trained by OpenAI. Respond conversationally",
        help="Base prompt for chatbot",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=False,
        required=True,
        help="Path to config yaml file",
    )
    parser.add_argument(
        "--submit_key",
        type=str,
        default=None,
        help="Custom submit key for chatbot. For more information on keys, see README",
    )
    parser.add_argument(
        "--no_stream",
        action="store_true",
        help="Disable streaming",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=False,
        help="image path",
    )
    args = parser.parse_args()

    # Initialize chatbot
    if os.path.exists(args.config):
        chatbot = ChatbotCLI(
            temperature=args.temperature,
            top_p=args.top_p,
            system_prompt=args.system_prompt,
        )
        try:
            chatbot.load(args.config)
        except Exception:
            print(f"Error: {args.config} could not be loaded")
            sys.exit()
    else:
        print(f"Error: {args.config} do not exists")
        sys.exit()

    session = create_session()
    completer = create_completer(
        [
            "!help",
            "!exit",
            "!reset",
            "!switch",
            "!rollback",
            "!save",
            "!load",
            "!config",
            "!openai",
            "!temperture",
            "!top_p",
        ],
    )
    key_bindings = create_keybindings()
    if args.submit_key:
        key_bindings = create_keybindings(args.submit_key)
    # Start chat

    while True:
        print()
        try:
            print("User: ")
            prompt = get_input(
                session=session,
                completer=completer,
                key_bindings=key_bindings,
            )
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit()
        if prompt.startswith("!"):
            try:
                chatbot.handle_commands(prompt)
            except Exception as e:
                print(f"Error: {e}")
            continue

        print(chatbot.engine + " ChatGPT: ", flush=True)
        if args.no_stream:
            print(chatbot.ask(prompt, "user"))
        else:
            if args.image and "[image]" in prompt:
                for query in chatbot.ask_stream(prompt,image=args.image):
                    print(query, end="", flush=True)
            else:
                for query in chatbot.ask_stream(prompt):
                    print(query, end="", flush=True)
        print()
