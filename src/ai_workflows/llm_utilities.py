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
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import json
import os
from jsonschema import validate, ValidationError, SchemaError
from langchain_aws import ChatBedrock
from langchain_anthropic import ChatAnthropic
import re


class LLMInterface:
    """Utility class for interacting with LLMs in AI workflows."""

    # class-level member variables
    temperature: float
    total_response_timeout_seconds: int
    number_of_retries: int
    seconds_between_retries: int
    llm: ChatOpenAI | AzureChatOpenAI | None
    model: str = ""
    json_retries: int = 2

    def __init__(self, openai_api_key: str = None, openai_model: str = None, temperature: float = 0.0,
                 total_response_timeout_seconds: int = 600, number_of_retries: int = 2,
                 seconds_between_retries: int = 5, azure_api_key: str = None, azure_api_engine: str = None,
                 azure_api_base: str = None, azure_api_version: str = None, langsmith_api_key: str = None,
                 langsmith_project: str = 'ai_workflows', langsmith_endpoint: str = 'https://api.smith.langchain.com',
                 json_retries: int = 2, anthropic_api_key: str = None, anthropic_model: str = None,
                 bedrock_model: str = None, bedrock_region: str = "us-east-1", bedrock_aws_profile: str = None):
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
        """

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
        self.json_retries = json_retries

        # initialize LangChain LLM access
        if openai_api_key:
            self.llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=temperature, model_name=openai_model,
                                  model_kwargs={"response_format": {"type": "json_object"}})
            # assume model is the model name for OpenAI
            self.model = openai_model
        elif azure_api_key:
            self.llm = AzureChatOpenAI(openai_api_key=azure_api_key, temperature=temperature,
                                       deployment_name=azure_api_engine, azure_endpoint=azure_api_base,
                                       openai_api_version=azure_api_version, openai_api_type="azure",
                                       model_kwargs={"response_format": {"type": "json_object"}})
            # assume model is the engine name for Azure
            self.model = azure_api_engine
        elif bedrock_model:
            if bedrock_aws_profile:
                # if we have an AWS profile, use it to configure Bedrock
                self.llm = ChatBedrock(credentials_profile_name=bedrock_aws_profile, model_id=bedrock_model,
                                       region_name=bedrock_region, model_kwargs={"temperature": temperature})
            else:
                # otherwise, assume that credentials are in AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and
                # AWS_SESSION_TOKEN environment variables
                self.llm = ChatBedrock(model_id=bedrock_model, region_name=bedrock_region,
                                       model_kwargs={"temperature": temperature})
            # assume model is the model name for OpenAI
            self.model = bedrock_model
        elif anthropic_api_key:
            self.llm = ChatAnthropic(api_key=anthropic_api_key, model=anthropic_model, temperature=temperature)
            # assume model is the model name for Anthropic
            self.model = anthropic_model
        else:
            raise ValueError("Must supply either OpenAI, Azure, Anthropic, or Bedrock parameters for LLM access.")

    def llm_json_response(self, prompt: str | list, json_validation_schema: str = "") -> tuple[dict | None, str, str]:
        """
        Call out to LLM for structured JSON response.

        This function sends a prompt to the LLM and returns the response in JSON format.

        :param prompt: Prompt to send to the LLM.
        :type prompt: str | list
        :param json_validation_schema: JSON schema for validating the JSON response (optional). Default is "", which
          means no validation.
        :type json_validation_schema: str
        :return: Tuple with parsed JSON response, raw LLM response, and error message (if any).
        :rtype: tuple(dict | None, str, str)
        """

        # execute LLM evaluation, but catch and return any exceptions
        try:
            # invoke LLM and parse+validate JSON response
            result = self.llm.invoke(prompt)
            json_objects = self.extract_json(result.content)
            validation_error = self._json_validation_error(json_objects, json_validation_schema)

            if validation_error and self.json_retries > 0:
                # if there was a validation error, retry up to the allowed number of times
                retries = 0
                while retries < self.json_retries:
                    if isinstance(prompt, str):
                        # if the prompt was a string, convert to list for the retry
                        retry_prompt = [HumanMessage(content=prompt)]
                    else:
                        # otherwise, make copy of the prompt list for retry
                        retry_prompt = prompt.copy()
                    # add original response
                    retry_prompt.append(AIMessage(content=result['raw'].content))
                    # add retry prompt, with or without a schema to guide the retry
                    if json_validation_schema:
                        retry_prompt.append(HumanMessage(content=f"Your JSON response was invalid. Please correct it "
                                                                 f"and respond with valid JSON (with no code block "
                                                                 f"or other content). Just 100% valid JSON, according "
                                                                 f"to the instructions given. Your JSON response "
                                                                 f"should match the following schema:\n\n"
                                                                 f"{json_validation_schema}\n\nYour JSON response:"))
                    else:
                        retry_prompt.append(HumanMessage(content=f"Your JSON response was invalid. Please correct it "
                                                                 f"and respond with valid JSON (with no code block "
                                                                 f"or other content). Just 100% valid JSON, according "
                                                                 f"to the instructions given:"))

                    # retry
                    result = self.llm.invoke(retry_prompt)
                    json_objects = self.extract_json(result.content)
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
        return json_objects[0] if not validation_error else None, result.content, validation_error

    def llm_json_response_with_timeout(self, prompt: str | list, json_validation_schema: str = "") -> dict | None:
        """
        Call out to LLM for structured JSON response with timeout and retry.

        This function sends a prompt to the LLM and returns the response in JSON format, with support for timeout and
        retry mechanisms.

        :param prompt: Prompt to send to the LLM.
        :type prompt: str | list
        :param json_validation_schema: JSON schema for validating the JSON response (optional). Default is "", which
          means no validation.
        :type json_validation_schema: str
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
        def _llm_json_response_with_timeout(inner_prompt: str | list, validation_schema: str = "") -> dict | None:
            try:
                # run async request on separate thread, wait for result with timeout and automatic retry
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.llm_json_response, inner_prompt, validation_schema)
                    result = future.result(timeout=self.total_response_timeout_seconds)
            except Exception as caught_e:
                # format error result like success result
                result = {"raw": BaseMessage(type="ERROR", content=f"{caught_e}")}
            return result

        return _llm_json_response_with_timeout(prompt, json_validation_schema)

    def generate_json_schema(self, json_output_spec: str) -> str:
        """
        Generate a JSON schema, adequate for JSON validation, based on a human-language JSON output specification.

        :param json_output_spec: Human-language JSON output specification.
        :type json_output_spec: str
        :return: JSON schema suitable for JSON validation purposes.
        :rtype: str
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

        # call out to LLM to generate JSON schema
        parsed_response, response, error = self.llm_json_response_with_timeout(json_schema_prompt, json_schema_schema)

        # raise error if any
        if error:
            raise RuntimeError(f"Failed to generate JSON schema: {error}")

        # return as nicely-formatted version of parsed response
        return json.dumps(parsed_response, indent=2)

    @staticmethod
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

        # return result, which could be an empty list if we didn't manage to find and parse anything
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
