"""
A simple wrapper for the official azure ChatGPT API
"""
import re
import argparse
import yaml
import os
import sys
from typing import NoReturn
import time
from itertools import cycle

import base64
from mimetypes import guess_type

from openai import AzureOpenAI, OpenAI
import boto3
import json
import tiktoken

from .utils import create_completer
from .utils import create_keybindings
from .utils import create_session
from .utils import get_filtered_keys_from_object
from .utils import get_input
from .utils import load_genes, find_genes_in_text


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
        max_tokens: dict = {"gpt-4-turbo": 6000, "gpt-4":4000,"gpt-4o":50000,"claude3_haiku":6000, "claude3_sonnet":6000, "deepseek-v3": 50000,"deepseek-r1": 50000},
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
        # self.api = AzureOpenAI(
        #     api_key=self.api_key,
        #     api_version=self.api_version,
        #     azure_endpoint=f"{self.api_base}"
        # )
        self.api = OpenAI(
            api_key=self.api_key,
            base_url=f"{self.api_base}"
        )
    

    def init_claude(self):
        self.api = boto3.client("bedrock-runtime", region_name="us-east-1")


    def add_to_conversation(
        self,
        message: str,
        role: str,
        # name: str = "",
        convo_id: str = "default",
        image: str|list = "",
    ) -> None:
        """
        Add a message to the conversation
        """
        # if image:
        #     if isinstance(image, list):
        #         content=[{"type": "text", "text": message}]
        #         for img in image:
        #             image_format = guess_type(img)[0]
        #             image_base64 = base64.b64encode(open(img, 'rb').read()).decode('utf-8')
        #             if 'claude3' in self.engine:
        #                 content += [
        #                         {"type": "image",
        #                             "source": {
        #                                 "type": "base64",
        #                                 "media_type": f"{image_format}",
        #                                 "data": f"{image_base64}",
        #                     }}]
        #             else:
        #                 content += [{"type": "image_url", "image_url": {"url": f"data:{image_format};base64,{image_base64}"}}]
        #     else:
        #         image_format = guess_type(image)[0]
        #         image_base64 = base64.b64encode(open(image, 'rb').read()).decode('utf-8')
        #         if 'claude3' in self.engine:
        #             content=[{"type": "text", "text": message}, 
        #                     {"type": "image",
        #                         "source": {
        #                             "type": "base64",
        #                             "media_type": f"{image_format}",
        #                             "data": f"{image_base64}",
        #                 }}]
        #         else:
        #             content=[{"type": "text", "text": message}, {"type": "image_url", "image_url": {"url": f"data:{image_format};base64,{image_base64}"}}]
        # else:
        content=[{"type": "text", "text": message}]

        self.conversation[convo_id].append({"role": role, "content": content})


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
        encoding = tiktoken.encoding_for_model("gpt-4")
        num_tokens = 0
        for message in self.conversation[convo_id]:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            if skip_system and message['role'] == "system":
                continue
            num_tokens += 3
            for key, value in message.items():
                if isinstance(value,list):
                    for content in value:
                        if "上传的文件名为" in content.get("text", "") and len(content.get("text", "")) < 50000:
                            continue
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

    def change_engine(self,engine):
        if engine in self.engine_list:
            self.__dict__.update(getattr(self, "engine_" + engine))
            if 'claude3' in engine:
                self.init_claude()
            else:
                self.init_openai()
        else:
            raise Exception("Wrong engine")

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
            try:
                # 阿里百炼的api限制, 如果content是list, 则需要将content转换为text
                if "deepseek" in self.model:
                    for message in self.conversation["current"]:
                        if isinstance(message.get("content"), list):
                            for content in message["content"]:
                                if content.get("text", "") != "":
                                    message["content"] = content["text"]
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
                if self.model == 'deepseek-r1':
                    content = "\n"+"="*20+"思考过程"+"="*20+"\n"
                    full_response += content
                    yield content
                is_answering = False
                for chunk in response:
                    if self.model == 'deepseek-r1':
                        if chunk.choices[0].delta.reasoning_content == "" and chunk.choices[0].delta.content == "":
                            pass
                        else:
                            # 如果思考结果为空，则开始打印完整回复
                            if chunk.choices[0].delta.reasoning_content is None and is_answering == False:
                                content = "\n"+"="*20+"完整回复"+"="*20+"\n"
                                full_response += content
                                yield content
                                # 防止打印多个“完整回复”标记
                                is_answering = True
                            # 如果思考过程不为空，则打印思考过程
                            if chunk.choices[0].delta.reasoning_content != "" and chunk.choices[0].delta.reasoning_content is not None:
                                content = chunk.choices[0].delta.reasoning_content
                                full_response += content
                                yield content
                            # 如果回复不为空，则打印回复。回复一般会在思考过程结束后返回
                            elif chunk.choices[0].delta.content != "" and chunk.choices[0].delta.content is not None:
                                content = chunk.choices[0].delta.content
                                full_response += content
                                yield content
                    else:
                        content = chunk.choices[0].delta.content
                        if content is not None:
                            full_response += content
                            yield content
            except:
                response = "服务器繁忙，请稍后重试。"
                for content in response:
                    yield content

        self.add_to_conversation(full_response, response_role, convo_id=convo_id)
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
    

    def get_recommend_question(
        self, 
        convo_id: str = "default",
        **kwargs,):
        if convo_id not in self.conversation:
            return "Please start a conversation first"
        recommend_prompt = '''任务：根据前面的对话,生成三个用户可能的后续提问,保证每个问题不超过20个字 
        结果输出为JSON格式:
    {{"questions": ["question1", "question2", "question3"]}}'''
        recommend_conversation_list = self.conversation[convo_id][-2:]
        recommend_conversation_list.append({"role": "user", "content": recommend_prompt})
        full_response = ""
        if 'claude3' in self.engine:
            return '{"questions": []}'
        else:
            response = self.api.chat.completions.create(
                messages=recommend_conversation_list,
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
                # response_format={ "type": "json_object" },
            )
            for resp in response:
                time.sleep(0.01)
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
        # print(recommend_conversation_list)
        return full_response


    def get_gene_disease_info(
        self, 
        convo_id: str = "default",
        genes_file_path: str ="",
        ):
        if convo_id not in self.conversation:
            return "Please start a conversation first"
        conversation_to_parse = str(self.conversation[convo_id][-2:])
        genes = load_genes(genes_file_path)
        # Find genes in the text
        found_genes = find_genes_in_text(conversation_to_parse, genes)
        
        # Output the found genes and their OMIM IDs
        for gene, omim_id in found_genes.items():
            print(f"{gene}: https://omim.org/entry/{omim_id}")


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
