import re
from typing import Set

from prompt_toolkit import prompt
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings

bindings = KeyBindings()


def create_keybindings(key: str = "c-@") -> KeyBindings:
    """
    Create keybindings for prompt_toolkit. Default key is ctrl+space.
    For possible keybindings, see: https://python-prompt-toolkit.readthedocs.io/en/stable/pages/advanced_topics/key_bindings.html#list-of-special-keys
    """

    @bindings.add(key)
    def _(event):
        event.app.exit(result=event.app.current_buffer.text)


    return bindings


def create_session() -> PromptSession:
    return PromptSession(history=InMemoryHistory())


def create_completer(commands: list, pattern_str: str = "$") -> WordCompleter:
    return WordCompleter(words=commands, pattern=re.compile(pattern_str))


def get_input(
    session: PromptSession = None,
    completer: WordCompleter = None,
    key_bindings=None,
) -> str:
    """
    Multiline input function.
    """
    return (
        session.prompt(
            completer=completer,
            multiline=True,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=key_bindings,
        )
        if session
        else prompt(multiline=True)
    )


async def get_input_async(
    session: PromptSession = None,
    completer: WordCompleter = None,
) -> str:
    """
    Multiline input function.
    """
    return (
        await session.prompt_async(
            completer=completer,
            multiline=True,
            auto_suggest=AutoSuggestFromHistory(),
        )
        if session
        else prompt(multiline=True)
    )


def get_filtered_keys_from_object(obj: object, *keys: str) -> Set[str]:
    """
    Get filtered list of object variable names.
    :param keys: List of keys to include. If the first key is "not", the remaining keys will be removed from the class keys.
    :return: List of class keys.
    """
    class_keys = obj.__dict__.keys()
    if not keys:
        return class_keys

    # Remove the passed keys from the class keys.
    if keys[0] == "not":
        return {key for key in class_keys if key not in keys[1:]}
    # Check if all passed keys are valid
    if invalid_keys := set(keys) - class_keys:
        raise ValueError(
            f"Invalid keys: {invalid_keys}",
        )
    # Only return specified keys that are in class_keys
    return {key for key in keys if key in class_keys}


def load_genes(file_path):
    genes = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            gene_name = parts[0].strip()
            omim_id = parts[1].strip()
            genes[gene_name] = omim_id
    return genes

def find_genes_in_text(text, genes):
    found_genes = {}
    for gene, omim_id in genes.items():
        # 使用正则表达式进行贪婪匹配，确保匹配整个基因名
        # 考虑到中文字符，使用 \b 可能不适用，改用更具体的边界匹配
        matches = re.findall(r'(?<![A-Za-z0-9])' + re.escape(gene) + r'(?![A-Za-z0-9])', text)
        if matches:
            found_genes[gene] = omim_id
    return found_genes