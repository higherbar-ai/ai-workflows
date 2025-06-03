#  Copyright (c) 2024 Higher Bar AI, PBC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Utilities for interacting with LLMs in AI workflows."""

from anthropic.types import Message, MessageParam, RawMessageStreamEvent
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI, AsyncStream
from openai import APITimeoutError, APIError, APIConnectionError, RateLimitError, InternalServerError
from anthropic import Anthropic, AsyncAnthropic, AnthropicBedrock, AsyncAnthropicBedrock
from anthropic import (APIConnectionError as AnthropicAPIConnectionError, RateLimitError as AnthropicRateLimitError,
                       InternalServerError as AnthropicInternalServerError, AsyncStream as AnthropicAsyncStream)
from langsmith import traceable, get_current_run_tree
from langsmith.wrappers import wrap_openai

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import json
import os
from jsonschema import validate, ValidationError, SchemaError
import re
from PIL import Image
import io
import base64
import json_repair
import tiktoken
import hashlib


class LLMInterface:
    """Utility class for interacting with LLMs in AI workflows."""

    # member variables
    temperature: float
    total_response_timeout_seconds: int
    number_of_retries: int
    seconds_between_retries: int
    llm: OpenAI | AzureOpenAI | Anthropic | AnthropicBedrock | None
    a_llm: AsyncOpenAI | AsyncAzureOpenAI | AsyncAnthropic | AsyncAnthropicBedrock | None
    model: str
    json_retries: int = 2
    max_tokens: int = 4096
    using_langsmith: bool = False
    system_prompt = ""
    maintain_history = False
    conversation_history: list = []

    def __init__(self, openai_api_key: str = None, openai_model: str = None, temperature: float = 0.0,
                 total_response_timeout_seconds: int = 600, number_of_retries: int = 2,
                 seconds_between_retries: int = 5, azure_api_key: str = None, azure_api_engine: str = None,
                 azure_api_base: str = None, azure_api_version: str = None, langsmith_api_key: str = None,
                 langsmith_project: str = 'ai_workflows', langsmith_endpoint: str = 'https://api.smith.langchain.com',
                 json_retries: int = 2, anthropic_api_key: str = None, anthropic_model: str = None,
                 bedrock_model: str = None, bedrock_region: str = "us-east-1", bedrock_aws_profile: str = None,
                 max_tokens: int = 4096, system_prompt: str = "", maintain_history: bool = False,
                 starting_chat_history: list[tuple] = None):
        """
        Initialize the LLM interface for LLM interactions.

        This function sets up the interface for interacting with various LLMs, including OpenAI and Azure, and
        configures the necessary parameters for API access and response handling.

        :param openai_api_key: OpenAI API key for accessing the LLM. Default is None.
        :type openai_api_key: str
        :param openai_model: OpenAI model name. Default is None.
        :type openai_model: str
        :param temperature: Temperature setting for the LLM. Default is 0.0.
        :type temperature: float
        :param total_response_timeout_seconds: Timeout for LLM responses in seconds. Default is 600.
        :type total_response_timeout_seconds: int
        :param number_of_retries: Number of retries for LLM calls. Default is 2.
        :type number_of_retries: int
        :param seconds_between_retries: Seconds between retries for LLM calls. Default is 5.
        :type seconds_between_retries: int
        :param azure_api_key: API key for Azure LLM. Default is None.
        :type azure_api_key: str
        :param azure_api_engine: Azure API engine name (deployment name; assumed to be the same as the OpenAI model
          name). Default is None.
        :type azure_api_engine: str
        :param azure_api_base: Azure API base URL. Default is None.
        :type azure_api_base: str
        :param azure_api_version: Azure API version. Default is None.
        :type azure_api_version: str
        :param langsmith_api_key: API key for LangSmith. Default is None.
        :type langsmith_api_key: str
        :param langsmith_project: LangSmith project name. Default is 'ai_workflows'.
        :type langsmith_project: str
        :param langsmith_endpoint: LangSmith endpoint URL. Default is 'https://api.smith.langchain.com'.
        :type langsmith_endpoint: str
        :param json_retries: Number of automatic retries for invalid JSON responses. Default is 2.
        :type json_retries: int
        :param anthropic_api_key: API key for Anthropic. Default is None.
        :type anthropic_api_key: str
        :param anthropic_model: Anthropic model name. Default is None.
        :type anthropic_model: str
        :param bedrock_model: AWS Bedrock model name. Default is None.
        :type bedrock_model: str
        :param bedrock_region: AWS Bedrock region. Default is "us-east-1".
        :type bedrock_region: str
        :param bedrock_aws_profile: AWS profile for Bedrock access. Default is None.
        :type bedrock_aws_profile: str
        :param max_tokens: Maximum tokens for LLM responses. Default is 4096.
        :type max_tokens: int
        :param system_prompt: System prompt to add to all LLM calls. Default is "".
        :type system_prompt: str
        :param maintain_history: Whether to maintain a history of LLM interactions (and include the history in each call
            to the LLM). Default is False.
        :type maintain_history: bool
        :param starting_chat_history: Starting chat history to use for the conversation chain (or None for none).
            Should be tuples, each with a human and an AI message.
        :type starting_chat_history: list[tuple]
        """

        # initialize LangSmith API (if key specified)
        if langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = langsmith_project
            os.environ["LANGCHAIN_ENDPOINT"] = langsmith_endpoint
            os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
            self.using_langsmith = True

        # configure model and request settings
        self.temperature = temperature
        self.total_response_timeout_seconds = total_response_timeout_seconds
        self.number_of_retries = number_of_retries
        self.seconds_between_retries = seconds_between_retries
        self.json_retries = json_retries
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.maintain_history = maintain_history

        # initialize LLM access
        if openai_api_key:
            self.llm = OpenAI(api_key=openai_api_key)
            self.a_llm = AsyncOpenAI(api_key=openai_api_key)
            if self.using_langsmith:
                # wrap OpenAI API with LangSmith for tracing
                self.llm = wrap_openai(self.llm)
                self.a_llm = wrap_openai(self.a_llm)
            self.model = openai_model
        elif azure_api_key:
            self.llm = AzureOpenAI(api_key=azure_api_key, azure_deployment=azure_api_engine,
                                   azure_endpoint=azure_api_base, api_version=azure_api_version)
            self.a_llm = AsyncAzureOpenAI(api_key=azure_api_key, azure_deployment=azure_api_engine,
                                          azure_endpoint=azure_api_base, api_version=azure_api_version)
            if self.using_langsmith:
                # wrap AzureOpenAI API with LangSmith for tracing
                self.llm = wrap_openai(self.llm, chat_name="ChatAzureOpenAI", completions_name="AzureOpenAI")
                self.a_llm = wrap_openai(self.a_llm, chat_name="ChatAzureOpenAI", completions_name="AzureOpenAI")
            # the Azure deployment name is the model name for Azure
            self.model = azure_api_engine
        elif bedrock_model:
            if bedrock_aws_profile:
                # if we have an AWS profile, use it to configure Bedrock
                self.llm = AnthropicBedrock(aws_profile=bedrock_aws_profile, aws_region=bedrock_region)
                self.a_llm = AsyncAnthropicBedrock(aws_profile=bedrock_aws_profile, aws_region=bedrock_region)
            else:
                # otherwise, assume that credentials are in AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and
                # AWS_SESSION_TOKEN environment variables, or otherwise pre-configured via the CLI
                self.llm = AnthropicBedrock(aws_region=bedrock_region)
                self.a_llm = AsyncAnthropicBedrock(aws_region=bedrock_region)
            self.model = bedrock_model
        elif anthropic_api_key:
            self.llm = Anthropic(api_key=anthropic_api_key)
            self.a_llm = AsyncAnthropic(api_key=anthropic_api_key)
            self.model = anthropic_model
        else:
            raise ValueError("Must supply either OpenAI, Azure, Anthropic, or Bedrock parameters for LLM access.")

        # initialize chat history
        if maintain_history:
            self.conversation_history = []
            if starting_chat_history is not None:
                for (human_message, ai_message) in starting_chat_history:
                    self.conversation_history += [
                        self.user_message(human_message),
                        self.ai_message(ai_message)
                    ]

    @traceable(run_type="prompt", name="ai_workflows.get_json_response")
    def get_json_response(self, prompt: str | list, json_validation_schema: str = "",
                          json_validation_desc: str = "",
                          bypass_history_and_system_prompt=False) -> tuple[dict | None, str, str]:
        """
        Call out to LLM for structured JSON response (synchronous version).

        This function sends a prompt to the LLM and returns the response in JSON format.

        :param prompt: Prompt to send to the LLM.
        :type prompt: str | list
        :param json_validation_schema: JSON schema for validating the JSON response (optional). Default is "".
        :type json_validation_schema: str
        :param json_validation_desc: Description of the JSON schema for validating response (optional). Default is "".
            If supplied, will be converted into a JSON schema, cached in-memory, and used for validation.
        :param bypass_history_and_system_prompt: Whether to bypass the history and system prompt. Default is False.
        :type bypass_history_and_system_prompt: bool
        :return: Tuple with parsed JSON response, raw LLM response, and error message (if any).
        :rtype: tuple(dict | None, str, str)
        """

        # execute LLM evaluation, but catch and return any exceptions
        try:
            # if we have a JSON schema description, convert it to a schema and cache it
            if not json_validation_schema and json_validation_desc:
                # use cached schema if available
                json_validation_schema = JSONSchemaCache.get_json_schema(json_validation_desc)

                # if no cached version available, generate and cache it now
                if not json_validation_schema:
                    json_validation_schema = self.generate_json_schema(json_validation_desc)
                    JSONSchemaCache.put_json_schema(json_validation_desc, json_validation_schema)

            # invoke LLM and parse+validate JSON response
            result = self.get_llm_response(prompt,
                                           bypass_history_and_system_prompt=bypass_history_and_system_prompt,
                                           json_mode=True)
            json_objects = self.extract_json(result)
            validation_error = self._json_validation_error(json_objects, json_validation_schema)

            if validation_error and self.json_retries > 0:
                # if there was a validation error, retry up to the allowed number of times
                retries = 0
                while retries < self.json_retries:
                    # draft our retry prompt
                    if json_validation_schema:
                        retry_prompt_text = f"Your JSON response was invalid. Please correct it " \
                                            f"and respond with valid JSON (with no code block " \
                                            f"or other content). Just 100% valid JSON, according " \
                                            f"to the instructions given. Your JSON response " \
                                            f"should match the following schema:\n\n" \
                                            f"{json_validation_schema}\n\nYour JSON response:"
                    else:
                        retry_prompt_text = "Your JSON response was invalid. Please correct it " \
                                            "and respond with valid JSON (with no code block " \
                                            "or other content). Just 100% valid JSON, according " \
                                            "to the instructions given.\n\nYour JSON response:"

                    # if we're maintaining history already, can just use the text as the prompt
                    if self.maintain_history and not bypass_history_and_system_prompt:
                        retry_prompt = retry_prompt_text
                    else:
                        # otherwise, we need to manually add the relevant history, starting with the current prompt
                        if isinstance(prompt, str):
                            # if the prompt was a string, convert to list for the retry
                            retry_prompt = [self.user_message(prompt)]
                        else:
                            # otherwise, make copy of the prompt list for retry
                            retry_prompt = prompt.copy()
                        # next add the original response
                        retry_prompt.append(self.ai_message(result))
                        # finally, add the retry prompt text
                        retry_prompt.append(self.user_message(retry_prompt_text))

                    # execute the retry
                    result = self.get_llm_response(retry_prompt,
                                                   bypass_history_and_system_prompt=bypass_history_and_system_prompt,
                                                   json_mode=True)
                    json_objects = self.extract_json(result)
                    validation_error = self._json_validation_error(json_objects, json_validation_schema)
                    retries += 1

                    # break if we got a valid response, otherwise keep going till we run out of retries
                    if not validation_error:
                        break

            if validation_error:
                # if we're out of retries and still have a validation error, we'll fall through to return it
                pass
        except Exception as caught_e:
            # catch and return the error with no response
            return None, "", str(caught_e)

        # return result
        return json_objects[0] if not validation_error else None, result, validation_error

    @traceable(run_type="prompt", name="ai_workflows.a_get_json_response")
    async def a_get_json_response(self, prompt: str | list,
                                  json_validation_schema: str = "",
                                  bypass_history_and_system_prompt=False) -> tuple[dict | None, str, str]:
        """
        Call out to LLM for structured JSON response (async version).

        This function sends a prompt to the LLM and returns the response in JSON format.

        :param prompt: Prompt to send to the LLM.
        :type prompt: str | list
        :param json_validation_schema: JSON schema for validating the JSON response (optional). Default is "", which
          means no validation.
        :type json_validation_schema: str
        :param bypass_history_and_system_prompt: Whether to bypass the history and system prompt. Default is False.
        :type bypass_history_and_system_prompt: bool
        :return: Tuple with parsed JSON response, raw LLM response, and error message (if any).
        :rtype: tuple(dict | None, str, str)
        """

        # execute LLM evaluation, but catch and return any exceptions
        try:
            # invoke LLM and parse+validate JSON response
            result = await self.a_get_llm_response(prompt,
                                                   bypass_history_and_system_prompt=bypass_history_and_system_prompt,
                                                   json_mode=True)
            json_objects = self.extract_json(result)
            validation_error = self._json_validation_error(json_objects, json_validation_schema)

            if validation_error and self.json_retries > 0:
                # if there was a validation error, retry up to the allowed number of times
                retries = 0
                while retries < self.json_retries:
                    # draft our retry prompt
                    if json_validation_schema:
                        retry_prompt_text = f"Your JSON response was invalid. Please correct it " \
                                            f"and respond with valid JSON (with no code block " \
                                            f"or other content). Just 100% valid JSON, according " \
                                            f"to the instructions given. Your JSON response " \
                                            f"should match the following schema:\n\n" \
                                            f"{json_validation_schema}\n\nYour JSON response:"
                    else:
                        retry_prompt_text = "Your JSON response was invalid. Please correct it " \
                                            "and respond with valid JSON (with no code block " \
                                            "or other content). Just 100% valid JSON, according " \
                                            "to the instructions given.\n\nYour JSON response:"

                    # if we're maintaining history already, can just use the text as the prompt
                    if self.maintain_history and not bypass_history_and_system_prompt:
                        retry_prompt = retry_prompt_text
                    else:
                        # otherwise, we need to manually add the relevant history, starting with the current prompt
                        if isinstance(prompt, str):
                            # if the prompt was a string, convert to list for the retry
                            retry_prompt = [self.user_message(prompt)]
                        else:
                            # otherwise, make copy of the prompt list for retry
                            retry_prompt = prompt.copy()
                        # next add the original response
                        retry_prompt.append(self.ai_message(result))
                        # finally, add the retry prompt text
                        retry_prompt.append(self.user_message(retry_prompt_text))

                    # execute the retry
                    result = await self.a_get_llm_response(
                        retry_prompt, bypass_history_and_system_prompt=bypass_history_and_system_prompt,
                        json_mode=True)
                    json_objects = self.extract_json(result)
                    validation_error = self._json_validation_error(json_objects, json_validation_schema)
                    retries += 1

                    # break if we got a valid response, otherwise keep going till we run out of retries
                    if not validation_error:
                        break

            if validation_error:
                # if we're out of retries and still have a validation error, we'll fall through to return it
                pass
        except Exception as caught_e:
            # catch and return the error with no response
            return None, "", str(caught_e)

        # return result
        return json_objects[0] if not validation_error else None, result, validation_error

    @traceable(run_type="chain", name="ai_workflows.get_llm_response")
    def get_llm_response(self, prompt: str | list, bypass_history_and_system_prompt=False,
                         json_mode: bool = False) -> str:
        """
        Call out to LLM for a response to a prompt (synchronous version).

        :param prompt: Prompt to send to the LLM (simple string or list of user and assistant messages).
        :type prompt: str | list
        :param bypass_history_and_system_prompt: Whether to bypass the history and system prompt. Default is False.
        :type bypass_history_and_system_prompt: bool
        :param json_mode: Whether to use model's "JSON mode" (if available). Default is False.
        :type json_mode: bool
        :return: Content of the LLM response.
        :rtype: str
        """

        # if prompt is a string, convert to message list for consistency
        if isinstance(prompt, str):
            prompt = [self.user_message(prompt)]

        # construct a full prompt with history if we're maintaining it
        if self.maintain_history and not bypass_history_and_system_prompt:
            prompt_with_history = self.conversation_history + prompt
        else:
            prompt_with_history = prompt

        # execute LLM evaluation, with appropriate parameters, depending on the LLM type
        if isinstance(self.llm, OpenAI) or isinstance(self.llm, AzureOpenAI):
            if json_mode and self.model == "o1-preview":
                # (o1-preview doesn't support JSON mode)
                json_mode = False

            result = self._llm_call(model=self.model, messages=prompt_with_history, max_tokens=self.max_tokens,
                                    temperature=self.temperature,
                                    response_format={"type": "json_object"} if json_mode else None,
                                    no_system_prompt=bypass_history_and_system_prompt)
            # extract the content from the response
            retval = result.choices[0].message.content
        elif isinstance(self.llm, Anthropic) or isinstance(self.llm, AnthropicBedrock):
            result = self._llm_call(model=self.model, messages=prompt_with_history, max_tokens=self.max_tokens,
                                    temperature=self.temperature, no_system_prompt=bypass_history_and_system_prompt)
            # merge response data together
            retval = ''.join(block.text for block in result.content)
        else:
            raise ValueError("LLM type not recognized.")

        # if we're maintaining history, add prompt and response
        if self.maintain_history and not bypass_history_and_system_prompt:
            self.conversation_history += prompt + [self.ai_message(retval)]

        # return the content of the LLM response
        return retval

    @traceable(run_type="chain", name="ai_workflows.a_get_llm_response")
    async def a_get_llm_response(self, prompt: str | list, bypass_history_and_system_prompt=False,
                                 json_mode: bool = False) -> str:
        """
        Call out to LLM for a response to a prompt (async version).

        :param prompt: Prompt to send to the LLM (simple string or list of user and assistant messages).
        :type prompt: str | list
        :param bypass_history_and_system_prompt: Whether to bypass the history and system prompt. Default is False.
        :type bypass_history_and_system_prompt: bool
        :param json_mode: Whether to use model's "JSON mode" (if available). Default is False.
        :type json_mode: bool
        :return: Content of the LLM response.
        :rtype: str
        """

        # if prompt is a string, convert to message list for consistency
        if isinstance(prompt, str):
            prompt = [self.user_message(prompt)]

        # construct a full prompt with history if we're maintaining it
        if self.maintain_history and not bypass_history_and_system_prompt:
            prompt_with_history = self.conversation_history + prompt
        else:
            prompt_with_history = prompt

        # execute LLM evaluation, with appropriate parameters, depending on the LLM type
        if isinstance(self.a_llm, AsyncOpenAI) or isinstance(self.a_llm, AsyncAzureOpenAI):
            if json_mode and self.model == "o1-preview":
                # (o1-preview doesn't support JSON mode)
                json_mode = False

            result = await self._a_llm_call(model=self.model, messages=prompt_with_history, max_tokens=self.max_tokens,
                                            temperature=self.temperature,
                                            response_format={"type": "json_object"} if json_mode else None,
                                            no_system_prompt=bypass_history_and_system_prompt)
            # extract the content from the response
            retval = result.choices[0].message.content
        elif isinstance(self.a_llm, AsyncAnthropic) or isinstance(self.a_llm, AsyncAnthropicBedrock):
            result = await self._a_llm_call(model=self.model, messages=prompt_with_history,
                                            max_tokens=self.max_tokens, temperature=self.temperature,
                                            no_system_prompt=bypass_history_and_system_prompt)
            # merge response data together
            retval = ''.join(block.text for block in result.content)
        else:
            raise ValueError("LLM type not recognized.")

        # if we're maintaining history, add prompt and response
        if self.maintain_history and not bypass_history_and_system_prompt:
            self.conversation_history += prompt + [self.ai_message(retval)]

        # return the content of the LLM response
        return retval

    def _llm_call(self, *args, **kwargs) -> (ChatCompletion | AsyncStream[ChatCompletionChunk] | Message |
                                             AnthropicAsyncStream[RawMessageStreamEvent]):
        """
        Internal wrapper function to call OpenAI's create method with an open-ended range of args and kwargs
        (synchronous version), with timeout and retries.

        :param args: Positional arguments to pass to the create method.
        :param kwargs: Keyword arguments to pass to the create method. Include 'no_system_prompt' as False if you
            want to suppress automatic addition of the configured system prompt.
        :return: Result of the create method call.
        """

        # define the retry decorator inside the method (so that we can use instance variables)
        retry_decorator = retry(
            stop=stop_after_attempt(self.number_of_retries),
            wait=wait_fixed(self.seconds_between_retries),
            retry=retry_if_exception_type((APITimeoutError, APIError, APIConnectionError, RateLimitError,
                                           InternalServerError, AnthropicAPIConnectionError, AnthropicRateLimitError,
                                           AnthropicInternalServerError)),
            reraise=True
        )

        @retry_decorator
        @traceable(run_type="llm", name="ai_workflows._llm_call_inner")
        def _llm_call_inner(*iargs, **ikwargs) -> (ChatCompletion | AsyncStream[ChatCompletionChunk] | Message |
                                                   AnthropicAsyncStream[RawMessageStreamEvent]):
            if isinstance(self.llm, OpenAI) or isinstance(self.llm, AzureOpenAI):
                return self.llm.chat.completions.create(*iargs, **ikwargs)
            elif isinstance(self.llm, Anthropic) or isinstance(self.llm, AnthropicBedrock):
                # need to manually add tracing details for Anthropic (not for OpenAI since it's wrapped)
                # grab the run tree and add starting metadata
                rt = get_current_run_tree()
                rt.metadata["model"] = ikwargs["model"]
                rt.metadata["max_tokens"] = ikwargs["max_tokens"]
                rt.metadata["temperature"] = ikwargs["temperature"]
                # call Anthropic
                inner_result = self.llm.messages.create(*iargs, **ikwargs)
                # add usage metadata to the run tree (but it won't show nicely in the UI)
                # (to show it nicely, we'd need to return a dict with a usage key and the content)
                rt.add_metadata({
                    "usage_metadata": {
                        "prompt_tokens": inner_result.usage.input_tokens,
                        "completion_tokens": inner_result.usage.output_tokens,
                        "total_tokens": inner_result.usage.input_tokens + inner_result.usage.output_tokens
                    }})
                return inner_result
            else:
                raise ValueError("LLM type not recognized.")

        # add timeout to kwargs
        kwargs['timeout'] = self.total_response_timeout_seconds

        # adjust max_tokens for OpenAI o1 models
        if ((isinstance(self.llm, OpenAI) or isinstance(self.llm, AzureOpenAI))
                and 'o1' in self.model and 'max_tokens' in kwargs):
            kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')

        # handle no_system_prompt flag
        no_system_prompt = False
        if 'no_system_prompt' in kwargs:
            no_system_prompt = kwargs.pop('no_system_prompt')

        # execute call, adding system prompt if present and desired
        if isinstance(self.llm, OpenAI) or isinstance(self.llm, AzureOpenAI):
            # prepend system prompt to messages, if present
            if self.system_prompt and not no_system_prompt and 'messages' in kwargs:
                kwargs['messages'] = [self.system_message(self.system_prompt)] + kwargs['messages']

            # automatically retry refusals
            for _ in range(self.number_of_retries):
                result = _llm_call_inner(*args, **kwargs)
                message = result.choices[0].message
                if hasattr(message, 'refusal') and message.refusal:
                    continue
                return result
            raise RuntimeError(f"OpenAI refused to generate a response after {self.number_of_retries} retries")
        elif isinstance(self.llm, Anthropic) or isinstance(self.llm, AnthropicBedrock):
            # add system prompt to parameters, if we have one
            if self.system_prompt and not no_system_prompt:
                kwargs['system'] = self.system_prompt

            return _llm_call_inner(*args, **kwargs)
        else:
            raise ValueError("LLM type not recognized.")

    async def _a_llm_call(self, *args, **kwargs) \
            -> (ChatCompletion | AsyncStream[ChatCompletionChunk] | Message |
                AnthropicAsyncStream[RawMessageStreamEvent]):
        """
        Internal wrapper function to call OpenAI's create method with an open-ended range of args and kwargs (async
        version), with timeout and retries.

        :param args: Positional arguments to pass to the create method.
        :param kwargs: Keyword arguments to pass to the create method. Include 'no_system_prompt' as False if you
            want to suppress automatic addition of the configured system prompt.
        :return: Result of the create method call.
        """

        # define the retry decorator inside the method (so that we can use instance variables)
        retry_decorator = retry(
            stop=stop_after_attempt(self.number_of_retries),
            wait=wait_fixed(self.seconds_between_retries),
            retry=retry_if_exception_type((APITimeoutError, APIError, APIConnectionError, RateLimitError,
                                           InternalServerError, AnthropicAPIConnectionError, AnthropicRateLimitError,
                                           AnthropicInternalServerError)),
            reraise=True
        )

        @retry_decorator
        @traceable(run_type="llm", name="ai_workflows._a_llm_call_inner")
        async def _a_llm_call_inner(*iargs, **ikwargs) -> (ChatCompletion | AsyncStream[ChatCompletionChunk] | Message |
                                                           AnthropicAsyncStream[RawMessageStreamEvent]):
            if isinstance(self.a_llm, AsyncOpenAI) or isinstance(self.a_llm, AsyncAzureOpenAI):
                return await self.a_llm.chat.completions.create(*iargs, **ikwargs)
            elif isinstance(self.a_llm, AsyncAnthropic) or isinstance(self.a_llm, AsyncAnthropicBedrock):
                # need to manually add tracing details for Anthropic (not for OpenAI since it's wrapped)
                # grab the run tree and add starting metadata
                rt = get_current_run_tree()
                rt.metadata["model"] = ikwargs["model"]
                rt.metadata["max_tokens"] = ikwargs["max_tokens"]
                rt.metadata["temperature"] = ikwargs["temperature"]
                # call Anthropic
                inner_result = await self.a_llm.messages.create(*iargs, **ikwargs)
                # add usage metadata to the run tree (but it won't show nicely in the UI)
                # (to show it nicely, we'd need to return a dict with a usage key and the content)
                rt.add_metadata({
                    "usage_metadata": {
                        "prompt_tokens": inner_result.usage.input_tokens,
                        "completion_tokens": inner_result.usage.output_tokens,
                        "total_tokens": inner_result.usage.input_tokens + inner_result.usage.output_tokens
                    }})
                return inner_result
            else:
                raise ValueError("LLM type not recognized.")

        # add timeout to kwargs
        kwargs['timeout'] = self.total_response_timeout_seconds

        # adjust max_tokens for OpenAI o1 models
        if ((isinstance(self.llm, OpenAI) or isinstance(self.llm, AzureOpenAI))
                and 'o1' in self.model and 'max_tokens' in kwargs):
            kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')

        # handle no_system_prompt flag
        no_system_prompt = False
        if 'no_system_prompt' in kwargs:
            no_system_prompt = kwargs.pop('no_system_prompt')

        # add system prompt and refusal retries
        if isinstance(self.a_llm, AsyncOpenAI) or isinstance(self.a_llm, AsyncAzureOpenAI):
            # prepend system prompt to messages, if present
            if self.system_prompt and not no_system_prompt and 'messages' in kwargs:
                kwargs['messages'] = [self.system_message(self.system_prompt)] + kwargs['messages']

            # automatically retry refusals
            for _ in range(self.number_of_retries):
                result = await _a_llm_call_inner(*args, **kwargs)
                message = result.choices[0].message
                if hasattr(message, 'refusal') and message.refusal:
                    continue
                return result
            raise RuntimeError(f"OpenAI refused to generate a response after {self.number_of_retries} retries")
        elif isinstance(self.a_llm, AsyncAnthropic) or isinstance(self.a_llm, AsyncAnthropicBedrock):
            # add system prompt to parameters, if we have one
            if self.system_prompt and not no_system_prompt:
                kwargs['system'] = self.system_prompt

            return await _a_llm_call_inner(*args, **kwargs)
        else:
            raise ValueError("LLM type not recognized.")

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string (synchronous version).

        :param text: Text to count tokens in.
        :type text: str
        :return: Number of tokens in the text.
        :rtype: int
        """

        # use appropriate token-counting method depending on the LLM provider
        if isinstance(self.llm, OpenAI) or isinstance(self.llm, AzureOpenAI):
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        else:
            count = self.llm.messages.count_tokens(model=self.model,
                                                   messages=[MessageParam(role="user", content=text)])
            return count.input_tokens

    async def a_count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string (async version).

        :param text: Text to count tokens in.
        :type text: str
        :return: Number of tokens in the text.
        :rtype: int
        """

        # use appropriate token-counting method depending on the LLM provider
        if isinstance(self.a_llm, AsyncOpenAI) or isinstance(self.a_llm, AsyncAzureOpenAI):
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        else:
            count = await self.a_llm.beta.messages.count_tokens(model=self.model,
                                                                messages=[MessageParam(role="user", content=text)])
            return count.input_tokens

    def enforce_max_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate a string as necessary to fit within a maximum number of tokens (synchronous version).

        :param text: Text to potentially truncate.
        :type text: str
        :param max_tokens: Maximum number of tokens to allow.
        :type max_tokens: int
        :return: Original or truncated string.
        :rtype: str
        """

        # if text is already under the limit, return as-is
        ntokens = self.count_tokens(text)
        if ntokens <= max_tokens:
            return text

        # otherwise, truncate using different methods depending on the LLM provider
        if isinstance(self.llm, OpenAI) or isinstance(self.llm, AzureOpenAI):
            # for OpenAI, can convert to tokens, truncate, then convert back
            encoding = tiktoken.encoding_for_model(self.model)
            tokens = list(encoding.encode(text))[:max_tokens]
            return encoding.decode(tokens)
        else:
            # use binary search to find maximum length that fits within token limit (minimizing count_tokens() calls)
            low, high = 0, len(text)
            while low < high:
                mid = (low + high) // 2
                if self.count_tokens(text[:mid]) <= max_tokens:
                    low = mid + 1
                else:
                    high = mid

            # return truncated text
            return text[:low - 1]

    async def a_enforce_max_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate a string as necessary to fit within a maximum number of tokens (async version).

        :param text: Text to potentially truncate.
        :type text: str
        :param max_tokens: Maximum number of tokens to allow.
        :type max_tokens: int
        :return: Original or truncated string.
        :rtype: str
        """

        # if text is already under the limit, return as-is
        ntokens = self.count_tokens(text)
        if ntokens <= max_tokens:
            return text

        # otherwise, truncate using different methods depending on the LLM provider
        if isinstance(self.a_llm, AsyncOpenAI) or isinstance(self.a_llm, AsyncAzureOpenAI):
            # for OpenAI, can convert to tokens, truncate, then convert back
            encoding = tiktoken.encoding_for_model(self.model)
            tokens = list(encoding.encode(text))[:max_tokens]
            return encoding.decode(tokens)
        else:
            # use binary search to find maximum length that fits within token limit (minimizing count_tokens() calls)
            low, high = 0, len(text)
            while low < high:
                mid = (low + high) // 2
                if await self.a_count_tokens(text[:mid]) <= max_tokens:
                    low = mid + 1
                else:
                    high = mid

            # return truncated text
            return text[:low - 1]

    def reset_history(self):
        """
        Reset the conversation history.

        This function clears the conversation history maintained by the LLM interface.
        """

        self.conversation_history = []

    def system_message(self, system_message: str) -> dict:
        """
        Generate a system message.

        This function takes a system message and returns it in the message format expected by the LLM.

        :param system_message: System message to format.
        :type system_message: str
        :return: Formatted system message.
        :rtype: dict
        """

        # Anthropic system prompts are passed separately; here, we'll use the "user" role if it's an Anthropic model
        # will also use "user" role for o1-series models, until they add comprehensive support for the "developer" role
        retval = {
            "role": "user" if isinstance(self.llm, Anthropic) or isinstance(self.llm, AnthropicBedrock)
                              or "o1" in self.model else "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ]
        }

        # return system message
        return retval

    @staticmethod
    def ai_message(ai_message: str) -> dict:
        """
        Generate an AI message.

        This function takes an AI message and returns it in the message format expected by the LLM.

        :param ai_message: AI message to format.
        :type ai_message: str
        :return: Formatted AI message.
        :rtype: dict
        """

        # all LLMs use the same format
        retval = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": ai_message
                }
            ]
        }

        # return AI message
        return retval

    @staticmethod
    def user_message(user_message: str) -> dict:
        """
        Generate a user message.

        This function takes a user message and returns it in the message format expected by the LLM.

        :param user_message: User message to format.
        :type user_message: str
        :return: Formatted user message.
        :rtype: dict
        """

        # all LLMs use the same format
        retval = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_message
                }
            ]
        }

        # return user message
        return retval

    def user_message_with_image(self, user_message: str, image: Image.Image, max_bytes: int | None = None,
                                current_dpi: int | None = None) -> dict:
        """
        Generate a user message with an embedded image.

        This function takes a user message and an image and returns a combined message with the image embedded.

        :param user_message: User message to include with the image.
        :type user_message: str
        :param image: Image to embed in the message.
        :type image: Image.Image
        :param max_bytes: Maximum size in bytes for the image. If the image is larger than this, it will be resized to
            fit within the limit. Default is None, which means no limit. (If you specify a limit, you must also specify
            the current DPI of the image.)
        :type max_bytes: int | None
        :param current_dpi: Current DPI of the image. Defaults to None but must be set if max_bytes is specified.
        :type current_dpi: int | None
        :return: Combined message with the image embedded.
        :rtype: dict
        """

        # convert image to bytes
        image_bytes = self.get_image_bytes(image=image, output_format="PNG", max_bytes=max_bytes,
                                           current_dpi=current_dpi)

        # encode image bytes as base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # return appropriate message format depending on the LLM
        if isinstance(self.llm, OpenAI) or isinstance(self.llm, AzureOpenAI):
            # note that OpenAI requires the text to be first and the image second, otherwise it refuses the request
            retval = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_message
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    }
                ]
            }
        elif isinstance(self.llm, Anthropic) or isinstance(self.llm, AnthropicBedrock):
            retval = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_message
                    }
                ]
            }
        else:
            raise ValueError("LLM type not recognized.")

        # return combined message with image
        return retval

    @staticmethod
    def get_image_bytes(image: Image.Image, output_format: str = 'PNG', max_bytes: int | None = None,
                        current_dpi: int | None = None) -> bytes:
        """
        Convert a PIL Image to bytes in the specified format.

        This function takes a PIL Image object and converts it to a byte array in the specified format.

        :param image: PIL Image to convert.
        :type image: Image.Image
        :param output_format: Output format for the image (default is 'PNG').
        :type output_format: str
        :param max_bytes: Maximum size in bytes for the image. If the image is larger than this, it will be resized to
            fit within the limit. Default is None, which means no limit. (If you specify a limit, you must also specify
            the current DPI of the image.)
        :type max_bytes: int | None
        :param current_dpi: Current DPI of the image. Defaults to None but must be set if max_bytes is specified.
        :type current_dpi: int | None
        :return: Bytes representing the image in the specified format.
        :rtype: bytes
        """

        # validate parameters
        if max_bytes is not None and current_dpi is None:
            raise ValueError("If max_bytes is specified, current_dpi must also be specified.")

        dpi = current_dpi
        while True:
            # save the image with current DPI
            img_byte_arr = io.BytesIO()
            if max_bytes and dpi:
                save_kwargs = {
                    "format": output_format,
                    "optimize": True,
                    "dpi": (dpi, dpi)
                }
            else:
                save_kwargs = {
                    "format": output_format,
                    "optimize": True
                }
            image.save(img_byte_arr, **save_kwargs)
            current_bytes = img_byte_arr.getvalue()

            # if no max_bytes specified or size is under limit, return the bytes
            if max_bytes is None or len(current_bytes) <= max_bytes:
                return current_bytes

            # if image is too large, reduce DPI by 10%
            dpi = int(dpi * 0.9)

            # if DPI gets too low, raise an error
            if dpi < 50:  # 72 DPI is typically considered the minimum for screen display, so 50 would be quite low
                raise RuntimeError(
                    f"Unable to reduce image to meet size limit of {max_bytes} bytes. "
                    f"Current size: {len(current_bytes)} bytes"
                )

    async def a_generate_json_schema(self, json_output_spec: str) -> str:
        """
        Generate a JSON schema, adequate for JSON validation, based on a human-language JSON output specification
        (async version).

        :param json_output_spec: Human-language JSON output specification.
        :type json_output_spec: str
        :return: JSON schema suitable for JSON validation purposes.
        :rtype: str
        """

        # generate prompt and validation schema
        json_schema_prompt, json_schema_schema = self._get_schema_prompt_and_meta_schema(json_output_spec)

        # call out to LLM to generate JSON schema
        parsed_response, response, error = await self.a_get_json_response(json_schema_prompt, json_schema_schema,
                                                                          bypass_history_and_system_prompt=True)

        # raise error if any
        if error:
            raise RuntimeError(f"Failed to generate JSON schema: {error}")

        # return as nicely-formatted version of parsed response
        return json.dumps(parsed_response, indent=2)

    def generate_json_schema(self, json_output_spec: str) -> str:
        """
        Generate a JSON schema, adequate for JSON validation, based on a human-language JSON output specification
        (synchronous version).

        :param json_output_spec: Human-language JSON output specification.
        :type json_output_spec: str
        :return: JSON schema suitable for JSON validation purposes.
        :rtype: str
        """

        # generate prompt and validation schema
        json_schema_prompt, json_schema_schema = self._get_schema_prompt_and_meta_schema(json_output_spec)

        # call out to LLM to generate JSON schema
        parsed_response, response, error = self.get_json_response(json_schema_prompt, json_schema_schema,
                                                                  bypass_history_and_system_prompt=True)

        # raise error if any
        if error:
            raise RuntimeError(f"Failed to generate JSON schema: {error}")

        # return as nicely-formatted version of parsed response
        return json.dumps(parsed_response, indent=2)

    @staticmethod
    def _get_schema_prompt_and_meta_schema(json_output_spec: str) -> tuple[str, str]:
        """
        Get the prompt for generating a JSON schema and the JSON schema for validating the generated schema.

        :param json_output_spec: Human-language JSON output specification.
        :type json_output_spec: str
        :return: Tuple with the prompt for generating a JSON schema and the JSON schema for validating the generated
          schema.
        :rtype: tuple[str, str]
        """

        # create a prompt for the LLM to generate a JSON schema
        json_schema_prompt = f"""Please generate a JSON schema based on the following description. Ensure that the schema is valid according to JSON Schema Draft 7 and includes appropriate types, properties, and required fields. Output only the JSON Schema with no description, code blocks, or other content.

The description, within |@| delimiters:

|@|{json_output_spec}|@|

The JSON schema (and only the JSON schema) according to JSON Schema Draft 7:"""

        # set a meta-schema for validating returned JSON schema (from https://json-schema.org/draft-07/schema)
        json_schema_schema = """{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "http://json-schema.org/draft-07/schema#",
    "title": "Core schema meta-schema",
    "definitions": {
        "schemaArray": {
            "type": "array",
            "minItems": 1,
            "items": { "$ref": "#" }
        },
        "nonNegativeInteger": {
            "type": "integer",
            "minimum": 0
        },
        "nonNegativeIntegerDefault0": {
            "allOf": [
                { "$ref": "#/definitions/nonNegativeInteger" },
                { "default": 0 }
            ]
        },
        "simpleTypes": {
            "enum": [
                "array",
                "boolean",
                "integer",
                "null",
                "number",
                "object",
                "string"
            ]
        },
        "stringArray": {
            "type": "array",
            "items": { "type": "string" },
            "uniqueItems": true,
            "default": []
        }
    },
    "type": ["object", "boolean"],
    "properties": {
        "$id": {
            "type": "string",
            "format": "uri-reference"
        },
        "$schema": {
            "type": "string",
            "format": "uri"
        },
        "$ref": {
            "type": "string",
            "format": "uri-reference"
        },
        "$comment": {
            "type": "string"
        },
        "title": {
            "type": "string"
        },
        "description": {
            "type": "string"
        },
        "default": true,
        "readOnly": {
            "type": "boolean",
            "default": false
        },
        "writeOnly": {
            "type": "boolean",
            "default": false
        },
        "examples": {
            "type": "array",
            "items": true
        },
        "multipleOf": {
            "type": "number",
            "exclusiveMinimum": 0
        },
        "maximum": {
            "type": "number"
        },
        "exclusiveMaximum": {
            "type": "number"
        },
        "minimum": {
            "type": "number"
        },
        "exclusiveMinimum": {
            "type": "number"
        },
        "maxLength": { "$ref": "#/definitions/nonNegativeInteger" },
        "minLength": { "$ref": "#/definitions/nonNegativeIntegerDefault0" },
        "pattern": {
            "type": "string",
            "format": "regex"
        },
        "additionalItems": { "$ref": "#" },
        "items": {
            "anyOf": [
                { "$ref": "#" },
                { "$ref": "#/definitions/schemaArray" }
            ],
            "default": true
        },
        "maxItems": { "$ref": "#/definitions/nonNegativeInteger" },
        "minItems": { "$ref": "#/definitions/nonNegativeIntegerDefault0" },
        "uniqueItems": {
            "type": "boolean",
            "default": false
        },
        "contains": { "$ref": "#" },
        "maxProperties": { "$ref": "#/definitions/nonNegativeInteger" },
        "minProperties": { "$ref": "#/definitions/nonNegativeIntegerDefault0" },
        "required": { "$ref": "#/definitions/stringArray" },
        "additionalProperties": { "$ref": "#" },
        "definitions": {
            "type": "object",
            "additionalProperties": { "$ref": "#" },
            "default": {}
        },
        "properties": {
            "type": "object",
            "additionalProperties": { "$ref": "#" },
            "default": {}
        },
        "patternProperties": {
            "type": "object",
            "additionalProperties": { "$ref": "#" },
            "propertyNames": { "format": "regex" },
            "default": {}
        },
        "dependencies": {
            "type": "object",
            "additionalProperties": {
                "anyOf": [
                    { "$ref": "#" },
                    { "$ref": "#/definitions/stringArray" }
                ]
            }
        },
        "propertyNames": { "$ref": "#" },
        "const": true,
        "enum": {
            "type": "array",
            "items": true,
            "minItems": 1,
            "uniqueItems": true
        },
        "type": {
            "anyOf": [
                { "$ref": "#/definitions/simpleTypes" },
                {
                    "type": "array",
                    "items": { "$ref": "#/definitions/simpleTypes" },
                    "minItems": 1,
                    "uniqueItems": true
                }
            ]
        },
        "format": { "type": "string" },
        "contentMediaType": { "type": "string" },
        "contentEncoding": { "type": "string" },
        "if": { "$ref": "#" },
        "then": { "$ref": "#" },
        "else": { "$ref": "#" },
        "allOf": { "$ref": "#/definitions/schemaArray" },
        "anyOf": { "$ref": "#/definitions/schemaArray" },
        "oneOf": { "$ref": "#/definitions/schemaArray" },
        "not": { "$ref": "#" }
    },
    "default": true
}"""
        return json_schema_prompt, json_schema_schema

    @staticmethod
    @traceable(run_type="parser", name="ai_workflows.extract_json")
    def extract_json(text: str) -> list[dict]:
        """
        Extract JSON content from a string, handling various formats.

        :param text: Text containing potential JSON content.
        :type text: str
        :return: A list of extracted JSON objects, or [] if none could be parsed or found.
        :rtype: list[dict]
        """

        json_objects = []

        # first try: parse the entire text as JSON
        try:
            parsed = json.loads(text.strip())
            # if that worked, go ahead and return as a single JSON object
            return [parsed]
        except json.JSONDecodeError:
            # otherwise, continue to try other methods
            pass

        # second try: look for code blocks with an explicit json specification
        json_pattern = r"```json(.*?)```"
        json_matches = re.findall(json_pattern, text, re.DOTALL)

        if not json_matches:
            # third try: look for generic code blocks if no json-specific blocks found
            generic_pattern = r"```(.*?)```"
            generic_matches = re.findall(generic_pattern, text, re.DOTALL)

            for match in generic_matches:
                # skip if the match starts with a language specification other than 'json'
                if match.strip().split('\n')[0] in ['python', 'javascript', 'typescript', 'java', 'cpp', 'ruby']:
                    continue

                # otherwise, try to parse
                try:
                    parsed = json.loads(match.strip())
                    # if we could parse it, add it to our results
                    json_objects.append(parsed)
                except json.JSONDecodeError:
                    continue
        else:
            # process json-specific matches
            for match in json_matches:
                try:
                    parsed = json.loads(match.strip())
                    # if we could parse it, add it to our results
                    json_objects.append(parsed)
                except json.JSONDecodeError:
                    continue

        # if we still don't have any results, try to automatically repair the JSON
        if not json_objects:
            parsed = json_repair.repair_json(text.strip(), return_objects=True, ensure_ascii=False)
            if parsed and isinstance(parsed, dict):
                json_objects = [parsed]

        # return result, which could be an empty list if we didn't succeed in finding and parsing any JSON
        return json_objects

    @staticmethod
    def _json_validation_error(json_objects: list[dict], json_validation_schema: str = "") -> str:
        """
        Validate LLM-returned JSON, return error text if invalid.

        :param json_objects: JSON objects returned from the LLM.
        :type json_objects: list[dict]
        :param json_validation_schema: JSON schema for validating the JSON response (defaults to "" for no schema
          validation).
        :type json_validation_schema: str
        :return: "" if parsed JSON is valid, otherwise text of the validation error.
        :rtype: str
        """

        # check for parsing errors
        if not json_objects:
            return "JSON parsing error: no valid JSON found in response"
        elif len(json_objects) > 1:
            return f"JSON parsing error: {len(json_objects)} JSON objects found in response"
        elif json_validation_schema:
            # validate parsed JSON against schema
            parsed_json = json_objects[0]
            try:
                validate(instance=parsed_json, schema=json.loads(json_validation_schema))
            except json.JSONDecodeError as e:
                return f"Failed to parse JSON schema: {e}"
            except SchemaError as e:
                return f"JSON schema is invalid: {e}"
            except ValidationError as e:
                return f"JSON response is invalid: {e}"

        # if we made it this far, that means the JSON is valid
        return ""


class JSONSchemaCache:
    """Cache for JSON schemas."""

    # shared class-level member for schema cache
    schema_cache: dict[str, str] = {}

    @staticmethod
    def get_json_schema(json_description: str) -> str:
        """
        Retrieve cached schema from JSON description.

        :param json_description: Description of the JSON format.
        :type json_description: str
        :return: JSON schema or empty string if not found.
        :rtype: str
        """

        return JSONSchemaCache.schema_cache.get(JSONSchemaCache._get_description_hash(json_description), "")

    @staticmethod
    def put_json_schema(json_description: str, json_schema: str):
        """
        Cache a JSON schema.

        :param json_description: Description of the JSON format.
        :type json_description: str
        :param json_schema: JSON schema to cache.
        :type json_schema: str
        """

        JSONSchemaCache.schema_cache[JSONSchemaCache._get_description_hash(json_description)] = json_schema

    @staticmethod
    def _get_description_hash(description: str) -> str:
        """
        Get a hash of the description for use as a cache key.

        :param description: Description to hash.
        :type description: str
        :return: Hashed description.
        :rtype: str
        """

        normalized_description = ' '.join(description.split()).lower()
        return hashlib.sha256(normalized_description.encode('utf-8')).hexdigest()
