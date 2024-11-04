import os
from typing import Dict, Iterable, Union, Literal, Sequence, Collection, List
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe

import numpy as np


class TikTokenTokenizer:
  """
  Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
  """

  special_tokens: Dict[str, int]

  num_reserved_special_tokens = 256

  pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # pylint: disable=line-too-long

  def __init__(self, model_path: str, add_bos: bool, add_eos: bool):
    """
    Initializes the Tokenizer with a Tiktoken model.

    Args:
        model_path (str): The path to the Tiktoken model file.
    """

    mergeable_ranks = load_tiktoken_bpe(model_path)
    num_base_tokens = len(mergeable_ranks)
    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",  # end of turn
    ] + [f"<|reserved_special_token_{i}|>" for i in range(5, self.num_reserved_special_tokens - 5)]
    self.special_tokens = {token: num_base_tokens + i for i, token in enumerate(special_tokens)}
    self.model = tiktoken.Encoding(
        name=Path(model_path).name,
        pat_str=self.pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=self.special_tokens,
    )
    self.eos = add_eos
    self.bos = add_bos
    print(f"Reloaded tiktoken model from {model_path}")

    self.n_words: int = self.model.n_vocab
    # BOS / EOS token IDs
    self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
    self.eos_id: int = self.special_tokens["<|end_of_text|>"]
    self.pad_id: int = -1
    self.stop_tokens = {
        self.special_tokens["<|end_of_text|>"],
        self.special_tokens["<|eot_id|>"],
    }
    print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

  def encode(
      self,
      s: str,
      *,
      prefill_lengths: int,
      allowed_special: Union[Literal["all"], Collection[str]] = (),
      disallowed_special: Union[Literal["all"], Collection[str]] = (),
  ):
    """
    Encodes a string into a list of token IDs.

    Args:
        s (str): The input string to be encoded.
        bos (bool): Whether to prepend the beginning-of-sequence token.
        eos (bool): Whether to append the end-of-sequence token.
        allowed_tokens ("all"|set[str]): allowed special tokens in string
        disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string
        prefill_lengths

    Returns:
        list[int]: A list of token IDs.

    By default, setting disallowed_special=() encodes a string by ignoring
    special tokens. Specifically:
    - Setting `disallowed_special` to () will cause all text corresponding
      to special tokens to be encoded as natural text (insteading of raising
      an error).
    - Setting `allowed_special` to "all" will treat all text corresponding
      to special tokens to be encoded as special tokens.
    """
    assert isinstance(s, str)

    # The tiktoken tokenizer can handle <=400k chars without
    # pyo3_runtime.PanicException.
    TIKTOKEN_MAX_ENCODE_CHARS = 400_000

    # https://github.com/openai/tiktoken/issues/195
    # Here we iterate over subsequences and split if we exceed the limit
    # of max consecutive non-whitespace or whitespace characters.
    MAX_NO_WHITESPACES_CHARS = 25_000

    substrs = (
        substr
        for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
        for substr in self._split_whitespaces_or_nonwhitespaces(
            s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
        )
    )
    t: List[int] = []
    for substr in substrs:
      t.extend(
          self.model.encode(
              substr,
              allowed_special=set(allowed_special),
              disallowed_special=disallowed_special,
          )
      )

    if self.bos:
      t.insert(0, self.bos_id)
    if (len(t) < prefill_lengths):
      for _ in range(prefill_lengths-len(t)):
          t.append(self.pad_id)
    if self.eos:
      t.append(self.eos_id)
    return t,len(t)-2

  def decode(self, t) -> str:
    """
    Decodes a list of token IDs into a string.

    Args:
        t (List[int]): The list of token IDs to be decoded.

    Returns:
        str: The decoded string.
    """
    # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
    return self.model.decode(t)

  @staticmethod
  def _split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int):
    """
    Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
    consecutive whitespaces or consecutive non-whitespaces.
    """
    current_slice_len = 0
    current_slice_is_space = s[0].isspace() if len(s) > 0 else False
    slice_start = 0

    for i, _ in enumerate(s):
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


def build_tokenizer(tokenizer_path, add_bos, add_eos):
  """Loads the tokenizer at `tokenizer_path`"""
  print(f"Tokenizer path: {tokenizer_path}")
  return TikTokenTokenizer(tokenizer_path, add_bos, add_eos)


def get_tokenizer(tokenizer_path, add_bos, add_eos):
  # Load tokenizer
  tokenizer_model = build_tokenizer(tokenizer_path, add_bos, add_eos)
  return tokenizer_model


class TikTokenTest():

     def __init__(self):
         self.source_tokenizer =  get_tokenizer("./tokenizer_llama3.tiktoken", add_bos=False, add_eos=False)

     def test_tokenize(self):
         text = "This is a test"
         tokens = [2028, 374, 264, 1296]
         print(f" output tokens = {self.source_tokenizer.encode(text)}")
         assert(np.array_equal(self.source_tokenizer.encode(text), tokens))

if __name__ == "__main__":

    tokencls = TikTokenTest()
    tokencls.test_tokenize()
