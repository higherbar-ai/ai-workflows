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

from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import json
import os
import logging


class LLMInterface:
    """Utility class for interacting with LLMs in AI workflows."""

    # class-level member variables
    temperature: float
    total_response_timeout_seconds: int
    number_of_retries: int
    seconds_between_retries: int
    llm: ChatOpenAI | AzureChatOpenAI | None
    json_llm: Runnable | None
    model: str = ""

    def __init__(self, openai_api_key: str = None, openai_model: str = None, temperature: float = 0.0,
                 total_response_timeout_seconds: int = 600, number_of_retries: int = 2,
                 seconds_between_retries: int = 5, azure_api_key: str = None, azure_api_engine: str = None,
                 azure_api_base: str = None, azure_api_version: str = None, langsmith_api_key: str = None,
                 langsmith_project: str = 'ai_workflows', langsmith_endpoint: str = 'https://api.smith.langchain.com'):
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
        """

        # validate parameters
        if not openai_api_key and not azure_api_key:
            raise ValueError("Must supply either OpenAI or Azure parameters for LLM access.")

        # initialize LangSmith API (if key specified)
        if langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = langsmith_project
            os.environ["LANGCHAIN_ENDPOINT"] = langsmith_endpoint
            os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key

        # configure model and request settings
        self.temperature = temperature
        self.total_response_timeout_seconds = total_response_timeout_seconds
        self.number_of_retries = number_of_retries
        self.seconds_between_retries = seconds_between_retries

        # initialize LangChain LLM access
        if azure_api_key:
            self.llm = AzureChatOpenAI(openai_api_key=azure_api_key, temperature=temperature,
                                       deployment_name=azure_api_engine, azure_endpoint=azure_api_base,
                                       openai_api_version=azure_api_version, openai_api_type="azure")
            # assume model is the engine name for Azure
            self.model = azure_api_engine
        else:
            self.llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=temperature, model_name=openai_model)
            # assume model is the model name for OpenAI
            self.model = openai_model
        self.json_llm = self.llm.with_structured_output(method="json_mode", include_raw=True)

    def llm_json_response(self, prompt: str | list) -> dict | None:
        """
        Call out to LLM for structured JSON response.

        This function sends a prompt to the LLM and returns the response in JSON format.

        :param prompt: Prompt to send to the LLM.
        :type prompt: str | list
        :return: JSON response from the LLM (or None if no response).
        :rtype: dict
        """

        # execute LLM evaluation, but catch and return any exceptions
        try:
            result = self.json_llm.invoke(prompt)
        except Exception as caught_e:
            # format error result like success result
            result = {"raw": BaseMessage(type="ERROR", content=f"{caught_e}")}
        return result

    def llm_json_response_with_timeout(self, prompt: str | list) -> dict | None:
        """
        Call out to LLM for structured JSON response with timeout and retry.

        This function sends a prompt to the LLM and returns the response in JSON format, with support for timeout and
        retry mechanisms.

        :param prompt: Prompt to send to the LLM.
        :type prompt: str | list
        :return: JSON response from the LLM (or None if no response).
        :rtype: dict
        """

        # define the retry decorator inside the method (so that we can use instance variables)
        retry_decorator = retry(
            stop=stop_after_attempt(self.number_of_retries),
            wait=wait_fixed(self.seconds_between_retries),
            retry=retry_if_exception_type(concurrent.futures.TimeoutError),
            reraise=True
        )

        @retry_decorator
        def _llm_json_response_with_timeout(inner_prompt: str | list) -> dict | None:
            try:
                # run async request on separate thread, wait for result with timeout and automatic retry
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.llm_json_response, inner_prompt)
                    result = future.result(timeout=self.total_response_timeout_seconds)
            except Exception as caught_e:
                # format error result like success result
                result = {"raw": BaseMessage(type="ERROR", content=f"{caught_e}")}
            return result

        return _llm_json_response_with_timeout(prompt)

    @staticmethod
    def process_json_response(response: dict) -> tuple[str, dict]:
        """
        Process JSON response from LLM and return as raw response and parsed dictionary from JSON.

        This function processes the JSON response received from the LLM, handling errors and parsing the response as
        needed.

        :param response: JSON response from LLM.
        :type response: dict
        :return: Raw response and parsed dictionary from JSON.
        :rtype: tuple
        """

        parsed_response = None
        if response['raw'].type == "ERROR":
            # if we caught an error, report and save that error, then move on
            final_response = response['raw'].content
            logging.warning(f"Error from LLM: {final_response}")
        elif 'parsed' in response and response['parsed'] is not None:
            # if we got a parsed version, save the JSON version of that
            final_response = json.dumps(response['parsed'])
            parsed_response = response['parsed']
        elif 'parsing_error' in response and response['parsing_error'] is not None:
            # if there was a parsing error, report and save that error, then move on
            final_response = str(response['parsing_error'])
            logging.warning(f"Parsing error : {final_response}")
        else:
            final_response = ""
            logging.warning(f"Unknown response from LLM")

        # return response in both raw and parsed formats
        return final_response, parsed_response
