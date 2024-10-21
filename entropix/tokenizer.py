import os
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

import tiktoken
from tiktoken.load import load_tiktoken_bpe

logger = getLogger(__name__)



# The tiktoken tokenizer can handle <=400k chars without
# pyo3_runtime.PanicException.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# https://github.com/openai/tiktoken/issues/195
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000


class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]
    num_reserved_special_tokens = 256
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        assert os.path.isfile(model_path), model_path

        # Load the BPE (Byte Pair Encoding) ranks from the model file
        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)

        # Define special tokens
        special_tokens = [
            '<|begin_of_text|>',
            '<|end_of_text|>',
            '<|reserved_special_token_0|>',
            '<|reserved_special_token_1|>',
            '<|finetune_right_pad_id|>',
            '<|step_id|>',
            '<|start_header_id|>',
            '<|end_header_id|>',
            '<|eom_id|>',  # end of message
            '<|eot_id|>',  # end of turn
            '<|python_tag|>',
        ]
        reserved_tokens = [
            f'<|reserved_special_token_{2 + i}|>'
            for i in range(self.num_reserved_special_tokens - len(special_tokens))
        ]
        special_tokens = special_tokens + reserved_tokens

        # Assign IDs to special tokens
        self.special_tokens = {token: num_base_tokens + i for i, token in enumerate(special_tokens)}

        # Initialize the Tiktoken encoding
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        # Set up tokenizer attributes
        self.n_words: int = num_base_tokens + len(special_tokens)
        self.bos_id: int = self.special_tokens['<|begin_of_text|>']
        self.eos_id: int = self.special_tokens['<|end_of_text|>']
        self.eot_id: int = self.special_tokens['<|eot_id|>']
        self.eom_id: int = self.special_tokens['<|eom_id|>']
        self.python_tag_id = self.special_tokens['<|python_tag|>']
        self.pad_id: int = self.special_tokens['<|finetune_right_pad_id|>']
        self.stop_tokens = [
            self.special_tokens['<|eom_id|>'],
            self.special_tokens['<|eot_id|>'],
        ]

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal['all'], AbstractSet[str]]] = None,
        disallowed_special: Union[Literal['all'], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_special ("all"|set[str]): Allowed special tokens in the string.
            disallowed_special ("all"|set[str]): Special tokens that raise an error when in the string.

        Returns:
            list[int]: A list of token IDs.

        Note:
            Setting disallowed_special=() encodes a string by ignoring special tokens.
            Setting allowed_special="all" treats all text corresponding to special tokens as special tokens.
        """
        if allowed_special is None:
            allowed_special = set()
        assert isinstance(s, str)

        # Split the input string into substrings to handle long sequences
        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        
        # Encode each substring and combine the results
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        
        # Add beginning-of-sequence token if requested
        if bos:
            t.insert(0, self.bos_id)
        
        # Add end-of-sequence token if requested
        if eos:
            t.append(self.eos_id)
        
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (Sequence[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Cast the input to List[int] as required by Tiktoken
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.

        This method helps in handling very long sequences of whitespace or non-whitespace characters.

        Args:
            s (str): The input string to be split.
            max_consecutive_slice_len (int): Maximum length of consecutive whitespace or non-whitespace characters.

        Yields:
            str: Substrings of the input string.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]