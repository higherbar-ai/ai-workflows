============
ai_workflows
============

The ``ai_workflows`` package is a toolkit for supporting AI workflows (i.e., workflows that are pre-scripted and
repeatable, but utilize LLMs for various tasks). It's still in early development, but is ready to support piloting and
experimentation.

Installation
------------

Install the latest version with pip::

    pip install py-ai-workflows[docs]

If you don't need anything in the ``document_utilities`` module (relating to reading, parsing, and converting
documents), you can install a slimmed-down version with::

    pip install py-ai-workflows

Additional document-parsing dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you installed the full version with document-processing capabilities (``py-ai-workflows[docs]``), you'll also need
to install several other dependencies, which you can do by running the
`initial-setup.ipynb <https://github.com/higherbar-ai/ai-workflows/blob/main/src/initial-setup.ipynb>`_ Jupyter
notebook — or by installing them manually as follows.

First, download NTLK data for natural language text processing::

    # download NLTK data
    import nltk
    nltk.download('punkt', force=True)

Then install ``libreoffice`` for converting Office documents to PDF.

  On Linux::

    # install LibreOffice for document processing
    !apt-get install -y libreoffice

  On MacOS::

    # install LibreOffice for document processing
    brew install libreoffice

  On Windows::

    # install LibreOffice for document processing
    choco install -y libreoffice

AWS Bedrock support
^^^^^^^^^^^^^^^^^^^

Finally, if you're accessing models via AWS Bedrock, the AWS CLI needs to be installed and configured for AWS access.

Jupyter notebooks with Google Colab support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use `the colab-or-not package <https://github.com/higherbar-ai/colab-or-not>`_ to initialize a Jupyter notebook
for Google Colab or other environments::

    %pip install colab-or-not py-ai-workflows

    # download NLTK data
    import nltk
    nltk.download('punkt', force=True)

    # set up our notebook environment (including LibreOffice)
    from colab_or_not import NotebookBridge
    notebook_env = NotebookBridge(
        system_packages=["libreoffice"],
        config_path="~/.hbai/ai-workflows.env",
        config_template={
            "openai_api_key": "",
            "openai_model": "",
            "azure_api_key": "",
            "azure_api_base": "",
            "azure_api_engine": "",
            "azure_api_version": "",
            "anthropic_api_key": "",
            "anthropic_model": "",
            "langsmith_api_key": "",
        }
    )
    notebook_env.setup_environment()

Overview
---------

Here are the basics:

#. `The llm_utilities module <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.llm_utilities.html>`_ provides
   a simple interface for interacting with a large language model (LLM). It
   includes the ``LLMInterface`` class that can be used for executing individual workflow steps.
#. `The document_utilities module <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.document_utilities.html#>`_
   provides an interface for extracting Markdown-formatted text and structured data
   from various file formats. It includes functions for reading Word, PDF, Excel, CSV, HTML, and other file formats,
   and then converting them into Markdown or structured data for use in LLM interactions.
#. The `example-doc-conversion.ipynb <https://github.com/higherbar-ai/ai-workflows/blob/main/src/example-doc-conversion.ipynb>`_
   notebook provides a simple example of how to use the ``document_utilities``
   module to convert files to Markdown format, in either Google Colab or a local environment.
#. The `example-surveyeval-lite.ipynb <https://github.com/higherbar-ai/ai-workflows/blob/main/src/example-surveyeval-lite.ipynb>`_
   notebook provides a more realistic workflow example that uses the ``document_utilities`` module to convert a survey
   file to Markdown format and then to JSON format, and then uses the ``llm_utilities`` module to evaluate survey
   questions using an LLM. It also works in either Google Colab or a local environment.
#. The `example-testing.ipynb <https://github.com/higherbar-ai/ai-workflows/blob/main/src/example-testing.ipynb>`_
   notebook provides a basic set-up for testing Markdown conversion methods (LLM-assisted
   vs. not-LLM-assisted). At the moment, this notebook only works in a local environment.

Examples
^^^^^^^^

Converting a file to Markdown format (without LLM assistance)::

    from ai_workflows.document_utilities import DocumentInterface

    doc_interface = DocumentInterface()
    markdown = doc_interface.convert_to_markdown(file_path)

Converting a file to Markdown format (*with* LLM assistance)::

    from ai_workflows.llm_utilities import LLMInterface
    from ai_workflows.document_utilities import DocumentInterface

    llm_interface = LLMInterface(openai_api_key=openai_api_key)
    doc_interface = DocumentInterface(llm_interface=llm_interface)
    markdown = doc_interface.convert_to_markdown(file_path)

Converting a file to JSON format::

    from ai_workflows.llm_utilities import LLMInterface
    from ai_workflows.document_utilities import DocumentInterface

    llm_interface = LLMInterface(openai_api_key=openai_api_key)
    doc_interface = DocumentInterface(llm_interface=llm_interface)
    dict_list = doc_interface.convert_to_json(
        file_path,
        json_context = "The file contains a survey instrument with questions to be administered to rural Zimbabwean household heads by a trained enumerator.",
        json_job = "Your job is to extract questions and response options from the survey instrument.",
        json_output_spec = "Return correctly-formatted JSON with the following fields: ..."
    )

Converting a file to JSON format (with automatic JSON schema generation and response validation+retry)::

    from ai_workflows.llm_utilities import LLMInterface
    from ai_workflows.document_utilities import DocumentInterface

    llm_interface = LLMInterface(openai_api_key=openai_api_key)
    doc_interface = DocumentInterface(llm_interface=llm_interface)
    json_output_spec = "Return correctly-formatted JSON with the following fields: ..."
    json_output_schema = llm_interface.generate_json_schema(json_output_spec)
    dict_list = doc_interface.convert_to_json(
        file_path,
        json_context = "The file contains a survey instrument with questions to be administered to rural Zimbabwean household heads by a trained enumerator.",
        json_job = "Your job is to extract questions and response options from the survey instrument.",
        json_output_spec = json_output_spec,
        json_validation_schema = json_output_schema
    )

Converting a file to JSON format (with automatic JSON schema generation and response validation+retry) (plus in-memory
caching of JSON schemas so they aren't generated every time)::

    from ai_workflows.llm_utilities import LLMInterface, JSONSchemaCache
    from ai_workflows.document_utilities import DocumentInterface

    llm_interface = LLMInterface(openai_api_key=openai_api_key)
    doc_interface = DocumentInterface(llm_interface=llm_interface)
    json_output_spec = "Return correctly-formatted JSON with the following fields: ..."
    json_output_schema = JSONSchemaCache.get_json_schema(json_output_spec)
    if not json_output_schema:
        json_output_schema = llm_interface.generate_json_schema(json_output_spec)
        JSONSchemaCache.put_json_schema(json_output_spec, json_output_schema)
    dict_list = doc_interface.convert_to_json(
        file_path,
        json_context = "The file contains a survey instrument with questions to be administered to rural Zimbabwean household heads by a trained enumerator.",
        json_job = "Your job is to extract questions and response options from the survey instrument.",
        json_output_spec = json_output_spec,
        json_validation_schema = json_output_schema
    )

Technical notes
---------------

LLMInterface
^^^^^^^^^^^^

`The LLMInterface class <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.llm_utilities.html#ai_workflows.llm_utilities.LLMInterface>`_
provides a simple LLM interface with the following features:

#. Support for both OpenAI and Anthropic models, either directly or via Azure or AWS Bedrock

#. Support for both regular and JSON responses (using the LLM provider's "JSON mode" when possible)

#. Optional support for conversation history (tracking and automatic addition to each request)

#. Automatic validation of JSON responses against a formal JSON schema (with automatic retry to correct invalid JSON)

#. Automatic (LLM-based) generation of formal JSON schemas

#. Automatic timeouts for long-running requests

#. Automatic retry for failed requests (OpenAI refusals, timeouts, and other retry-able errors)

#. Support for LangSmith tracing

#. Synchronous and async versions of all functions (async versions begin with ``a_``)

Key methods:

#. `get_llm_response() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.llm_utilities.html#ai_workflows.llm_utilities.LLMInterface.get_llm_response>`_:
   Get a response from an LLM

#. `get_json_response() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.llm_utilities.html#ai_workflows.llm_utilities.LLMInterface.get_json_response>`_:
   Get a JSON response from an LLM

#. `user_message() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.llm_utilities.html#ai_workflows.llm_utilities.LLMInterface.user_message>`_:
   Get a properly-formatted user message to include in an LLM prompt

#. `user_message_with_image() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.llm_utilities.html#ai_workflows.llm_utilities.LLMInterface.user_message_with_image>`_:
   Get a properly-formatted user message to include in an LLM prompt, including an image
   attachment

#. `generate_json_schema() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.llm_utilities.html#ai_workflows.llm_utilities.LLMInterface.generate_json_schema>`_:
   Generate a JSON schema from a human-readable description

#. `count_tokens() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.llm_utilities.html#ai_workflows.llm_utilities.LLMInterface.count_tokens>`_:
   Count the number of tokens in a string

JSONSchemaCache
^^^^^^^^^^^^^^^

`The JSONSchemaCache class <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.llm_utilities.html#ai_workflows.llm_utilities.JSONSchemaCache>`_
provides a simple in-memory cache for JSON schemas, so that they don't have to be
regenerated repeatedly.

Key methods:

#. `get_json_schema() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.llm_utilities.html#ai_workflows.llm_utilities.JSONSchemaCache.get_json_schema>`_:
   Get a JSON schema from the cache (returns empty string if none found)

#. `put_json_schema() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.llm_utilities.html#ai_workflows.llm_utilities.JSONSchemaCache.put_json_schema>`_:
   Put a JSON schema into the cache

DocumentInterface
^^^^^^^^^^^^^^^^^

`The DocumentInterface class <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.document_utilities.html#ai_workflows.document_utilities.DocumentInterface>`_ provides a simple interface for converting files to Markdown or JSON format.

Key methods:

#. `convert_to_markdown() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.document_utilities.html#ai_workflows.document_utilities.DocumentInterface.convert_to_markdown>`_:
   Convert a file to Markdown format, using an LLM if available and deemed helpful (if you
   specify ``use_text=True``, it will include raw text in any LLM prompt, which might improve results)

#. `convert_to_json() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.document_utilities.html#ai_workflows.document_utilities.DocumentInterface.convert_to_json>`_:
   Convert a file to JSON format using an LLM (if you specify ``markdown_first=True``, it will
   convert to Markdown first, then convert to JSON; otherwise, it might convert to JSON page-by-page, using an LLM)

#. `markdown_to_json() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.document_utilities.html#ai_workflows.document_utilities.DocumentInterface.markdown_to_json>`_:
   Convert a Markdown string to JSON format using an LLM

#. `markdown_to_text() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.document_utilities.html#ai_workflows.document_utilities.DocumentInterface.markdown_to_text>`_:
   Convert a Markdown string to plain text

Markdown conversion
"""""""""""""""""""

The `DocumentInterface.convert_to_markdown() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.document_utilities.html#ai_workflows.document_utilities.DocumentInterface.convert_to_markdown>`_
method uses one of several methods to convert files to Markdown.

If an ``LLMInterface`` is available:

#. PDF files are converted to Markdown with LLM assistance: we split the PDF into pages (splitting double-page spreads
   as needed), convert each page to an image, and then convert to Markdown using the help of a multimodal LLM. This is
   the most accurate method, but it's also the most expensive, running at about $0.015 per page as of October 2024. In
   the process, we try to keep narrative text that flows across pages together, drop page headers and footers, and
   describe images, charts, and figures as if to a blind person. We also do our best to convert tables to proper
   Markdown tables. If the ``use_text`` parameter is set to ``True``, we'll extract the raw text from each page (when
   possible) and provide that to the LLM to assist it with the conversion.
#. We use LibreOffice to convert ``.docx``, ``.doc``, and ``.pptx`` files to PDF and then convert the PDF to Markdown
   using the LLM assistance method described above.
#. For ``.xlsx`` files without charts or images, we use a custom parser to convert worksheets and table ranges to proper
   Markdown tables. If there are charts or images, we use LibreOffice to convert to PDF and, if it's 10 pages or fewer,
   we convert from the PDF to Markdown using the LLM assistance method described above. If it's more than 10 pages,
   we fall back to dropping charts or images and converting without LLM assistance.
#. For other file types, we fall back to converting without LLM assistance, as described below.

Otherwise, we convert files to Markdown using one of the following methods (in order of preference):

#. For ``.xlsx`` files, we use a custom parser and Markdown formatter.
#. For other file types, we use IBM's ``Docling`` package for those file formats that it supports. This method drops
   images, charts, and figures, but it does a nice job with tables and automatically uses OCR when needed.
#. If ``Docling`` fails or doesn't support a file format, we next try ``PyMuPDFLLM``, which supports PDF files and a
   range of other formats. This method also drops images, charts, and figures, and it's pretty bad at tables, but it
   does a good job extracting text and a better job adding Markdown formatting than most other libraries.
#. Finally, if we haven't managed to convert the file using one of the higher-quality methods described above, we use
   the ``Unstructured`` library to parse the file into elements and then add basic Markdown formatting. This method is
   fast and cheap, but it's also the least accurate.

JSON conversion
"""""""""""""""

You can convert from Markdown to JSON using the `DocumentInterface.markdown_to_json() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.document_utilities.html#ai_workflows.document_utilities.DocumentInterface.markdown_to_json>`_
method, or you can convert files directly to JSON using the `DocumentInterface.convert_to_json() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.document_utilities.html#ai_workflows.document_utilities.DocumentInterface.convert_to_json>`_
method. The latter method will most often convert to Markdown first and then to JSON, but it will convert straight to
JSON with a page-by-page approach if:

#. The ``markdown_first`` parameter is explicitly provided as ``False`` and converting the file to Markdown would
   naturally use an LLM with a page-by-page approach (see the section above)
#. Or: converting the file to Markdown would naturally use an LLM with a page-by-page approach,
   the ``markdown_first`` parameter is not explicitly provided as ``True``, and the file's content doesn't look too
   large to fit in the LLM context window (<= 50 pages or 25,000 tokens).

The advantage of converting to JSON directly can also be a disadvantage: parsing to JSON is done page-by-page. If
structural elements don't span page boundaries, this can be great; however, if elements *do* span page boundaries, then
it will make things worse.

Whether or not you convert to JSON via Markdown, JSON conversion always uses LLM assistance. The parameters you supply
are:

#. ``json_context``: a description of the file's content, to help the LLM understand what it's looking at
#. ``json_job``: a description of the task you want the LLM to perform (e.g., extracting survey questions)
#. ``json_output_spec``: a description of the output you expect from the LLM
#. ``json_output_schema``: optionally, a formal JSON schema to validate the LLM's output; by
   default, this will be automatically generated based on your ``json_output_spec``, but you can specify your own
   schema or explicitly pass None if you want to disable JSON validation (if JSON validation isn't disabled, the
   ``LLMInterface`` default is to retry twice if the LLM output doesn't parse or match the schema, but you can change
   this behavior by specifying the ``json_retries`` parameter in the ``LLMInterface`` constructor)

The more detail you provide, the better the LLM will do at the JSON conversion.

If you find that things aren't working well, try including some few-shot examples in the ``json_output_spec`` parameter.

Known issues
^^^^^^^^^^^^

#. The example Google Colab notebooks pop up a message during installation that offers to restart the runtime. You have
   to click cancel so as not to interrupt execution.

#. The automatic generation and caching of JSON schemas (for response validation) can work poorly when batches of
   similar requests are all launched in parallel (as each request will generate and cache the schema).

#. LangSmith tracing support is imperfect in a few ways:

   a. For OpenAI models, the top-level token usage counts are roughly doubled. You have to look to the inner LLM call
      for an accurate count of input and output tokens.
   b. For Anthropic models, the token usage doesn't show up at all, but you can find it by clicking into the metadata
      for the inner LLM call.
   c. For Anthropic models, the system prompt is only visible if you click into the inner LLM call and then switch the
      *Input* display to *Raw input*.
   d. For Anthropic models, images in prompts don't show properly.

#. The support for conversation history in ``LLMInterface`` can overflow the context window in long conversations.

ImportError: libGL.so.1: cannot open shared object file
"""""""""""""""""""""""""""""""""""""""""""""""""""""""

If you use this package in a headless environment (e.g., within a Docker container), you might encounter the following
error::

    ImportError: libGL.so.1: cannot open shared object file: No such file or directory

This is caused by a conflict between how the Docling and Unstructured packages depend on opencv. The fix is to install
all of your requirements like normal, and then uninstall and re-install opencv::

    pip uninstall -y opencv-python opencv-python-headless && pip install opencv-python-headless

In a Dockerfile (after your ``pip install`` commands)::

    RUN pip uninstall -y opencv-python opencv-python-headless && pip install opencv-python-headless

Roadmap
-------

There's much that can be improved here. For example:

* For what's already here:
    * Adding unit tests
    * Tracking and reporting LLM costs
    * Improving evaluation and comparison methods
* Supporting more file formats and conversion methods:
    * Trying Claude's `direct PDF support <https://docs.anthropic.com/en/docs/build-with-claude/pdf-support>`_
* Expanding capabilities:
    * Adding support for logging workflow steps and results
    * Adding async versions of the ``DocumentInterface`` methods
    * Adding support for more LLMs
    * Adding support for a higher-level workflow-step concept that simplifies use of the ``LLMInterface`` and
      ``DocumentInterface`` classes
    * Adding basic RAG support
    * Expanding RAG support for knowledge graphs
    * Adding some kind of Docker support to extend the RAG/KG implementations to, e.g., ChatGPT via ChatGPT Actions
    * Adding automatic summarization of conversation histories to stay within a fixed token budget

Credits
-------

This toolkit was originally developed by `Higher Bar AI, PBC <https://higherbar.ai>`_, a public benefit corporation. To
contact us, email us at ``info@higherbar.ai``.

Many thanks also to `Laterite <https://www.laterite.com/>`_ for their contributions.

Full documentation
------------------

See the full reference documentation here:

    https://ai-workflows.readthedocs.io/

Local development
-----------------

To develop locally:

#. ``git clone https://github.com/higherbar-ai/ai-workflows``
#. ``cd ai-workflows``
#. ``python -m venv .venv``
#. ``source .venv/bin/activate``
#. ``pip install -e .``
#. Execute the ``initial-setup.ipynb`` Jupyter notebook to install system dependencies.

For convenience, the repo includes ``.idea`` project files for PyCharm.

To rebuild the documentation:

#. Update version number in ``/docs/source/conf.py``
#. Update layout or options as needed in ``/docs/source/index.rst``
#. In a terminal window, from the project directory:
    a. ``cd docs``
    b. ``SPHINX_APIDOC_OPTIONS=members,show-inheritance sphinx-apidoc -o source ../src/ai_workflows --separate --force``
    c. ``make clean html``

To rebuild the distribution packages:

#. For the PyPI package:
    a. Update version number (and any build options) in ``/setup.py``
    b. Confirm credentials and settings in ``~/.pypirc``
    c. Run ``/setup.py`` for the ``bdist_wheel`` and ``sdist`` build types (*Tools... Run setup.py task...* in PyCharm)
    d. Delete old builds from ``/dist``
    e. In a terminal window:
        i. ``twine upload dist/* --verbose``
#. For GitHub:
    a. Commit everything to GitHub and merge to ``main`` branch
    b. Add new release, linking to new tag like ``v#.#.#`` in main branch
#. For readthedocs.io:
    a. Go to https://readthedocs.org/projects/ai-workflows/, log in, and click to rebuild from GitHub (only if it
       doesn't automatically trigger)
