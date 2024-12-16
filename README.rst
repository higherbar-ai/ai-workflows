============
ai_workflows
============

The ``ai_workflows`` package is a toolkit for supporting AI workflows (i.e., workflows that are pre-scripted and
repeatable, but utilize LLMs for various tasks).

The goal is to lower the bar for social scientists and others to
leverage LLMs in repeatable, reliable, and transparent ways. See
`this blog post <https://www.linkedin.com/pulse/repeatable-reliable-transparent-graduating-from-ai-workflows-robert-nb4ge/>`_
for a discussion,
`here for the full documentation <https://ai-workflows.readthedocs.io/>`_, and
`here for a custom GPT <https://chatgpt.com/g/g-67586f2d154081918b6ee65b868e859e-ai-workflows-coding-assistant>`_
that can help you use this package. If you learn best by example, see these:

#. `example-doc-conversion.ipynb <https://github.com/higherbar-ai/ai-workflows/blob/main/src/example-doc-conversion.ipynb>`_:
   loading different file formats and converting them into a Markdown syntax that LLMs can understand.
#. `example-doc-extraction.ipynb <https://github.com/higherbar-ai/ai-workflows/blob/main/src/example-doc-extraction.ipynb>`_:
   extracting structured data from unstructured documents (edit notebook to customize).
#. `example-doc-extraction-templated.ipynb <https://github.com/higherbar-ai/ai-workflows/blob/main/src/example-doc-extraction-templated.ipynb>`_:
   extracting structured data from unstructured documents (supply an Excel template to customize).
#. `example-qual-analysis-1.ipynb <https://github.com/higherbar-ai/ai-workflows/blob/main/src/example-qual-analysis-1.ipynb>`_:
   a more realistic workflow example that performs a simple qualitative analysis on a set of interview transcripts.
#. `example-surveyeval-lite.ipynb <https://github.com/higherbar-ai/ai-workflows/blob/main/src/example-surveyeval-lite.ipynb>`_:
   another workflow example that critically evaluates surveys question-by-question.

Tip: if you're not completely comfortable working in Python, use
`GitHub Copilot in VS Code <https://code.visualstudio.com/docs/copilot/setup>`_
or Gemini as a copilot in `Google Colab <https://colab.google/>`_. Also do make use of
`this custom GPT coding assistant <https://chatgpt.com/g/g-67586f2d154081918b6ee65b868e859e-ai-workflows-coding-assistant>`_.

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
#. The `example-doc-extraction.ipynb <https://github.com/higherbar-ai/ai-workflows/blob/main/src/example-doc-extraction.ipynb>`_
   notebook provides an example of how to extract structured data from unstructured documents.
#. The `example-doc-extraction-templated.ipynb <https://github.com/higherbar-ai/ai-workflows/blob/main/src/example-doc-extraction-templated.ipynb>`_
   notebook provides an easier-to-customize version of the above: you supply an Excel template with your data extraction
   needs.
#. The `example-qual-analysis-1.ipynb <https://github.com/higherbar-ai/ai-workflows/blob/main/src/example-qual-analysis-1.ipynb>`_
   notebook provides a more realistic workflow example that uses both the ``document_utilities`` and the
   ``llm_utilities`` modules to perform a simple qualitative analysis on a set of interview transcripts. It also works
   in either Google Colab or a local environment.
#. The `example-surveyeval-lite.ipynb <https://github.com/higherbar-ai/ai-workflows/blob/main/src/example-surveyeval-lite.ipynb>`_
   notebook provides another workflow example that uses the ``document_utilities`` module to convert a survey
   file to Markdown format and then to JSON format, and then uses the ``llm_utilities`` module to evaluate survey
   questions using an LLM. It also works in either Google Colab or a local environment.
#. The `example-testing.ipynb <https://github.com/higherbar-ai/ai-workflows/blob/main/src/example-testing.ipynb>`_
   notebook provides a basic set-up for testing Markdown conversion methods (LLM-assisted
   vs. not-LLM-assisted). At the moment, this notebook only works in a local environment.

Example snippets
^^^^^^^^^^^^^^^^

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

Requesting a JSON response from an LLM::

    from ai_workflows.llm_utilities import LLMInterface

    llm_interface = LLMInterface(openai_api_key=openai_api_key)

    json_output_spec = """Return correctly-formatted JSON with the following fields:

    * `answer` (string): Your answer to the question."""

    full_prompt = f"""Answer the following question:

    (question)

    {json_output_spec}

    Your JSON response precisely following the instructions given:"""

    parsed_response, raw_response, error = llm_interface.get_json_response(
        prompt = full_prompt,
        json_validation_desc = json_output_spec
    )

Technical notes
---------------

Working with JSON
^^^^^^^^^^^^^^^^^

The ``ai_workflows`` package helps you to extract structured JSON content from documents and LLM responses. In all such
cases, you have to describe the JSON format that you want with enough clarity and specificity that the system can
reliably generate and validate responses (you typically supply this in a ``json_output_spec`` parameter). When describing
your desired JSON, always include the field names and types, as well as detailed descriptions. For example, if you
wanted a list of questions back::

    json_output_spec = """Return correctly-formatted JSON with the following fields:

    * `questions` (list of objects): A list of questions, each with the following fields:
        * `question` (string): The question text
        * `answer` (string): The supplied answer to the question"""

By default, the system will use this informal, human-readable description to automatically generate a formal JSON
schema, which it will use to validate LLM responses (and retry if needed).

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
   Generate a JSON schema from a human-readable description (called automatically when JSON output
   description is supplied to ``get_json_response()``)

#. `count_tokens() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.llm_utilities.html#ai_workflows.llm_utilities.LLMInterface.count_tokens>`_:
   Count the number of tokens in a string

#. `enforce_max_tokens() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.llm_utilities.html#ai_workflows.llm_utilities.LLMInterface.enforce_max_tokens>`_:
   Truncate a string as necessary to fit within a maximum number of tokens

If you don't have an API key for an AI provider yet, `see here to learn what that is and how to get one <https://www.linkedin.com/pulse/those-genai-api-keys-christopher-robert-l5rie/>`_.

DocumentInterface
^^^^^^^^^^^^^^^^^

`The DocumentInterface class <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.document_utilities.html#ai_workflows.document_utilities.DocumentInterface>`_ provides a simple interface for converting files to Markdown or JSON format.

Key methods:

#. `convert_to_markdown() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.document_utilities.html#ai_workflows.document_utilities.DocumentInterface.convert_to_markdown>`_:
   Convert a file to Markdown format, using an LLM if available and deemed helpful (if you
   specify ``use_text=True``, it will include raw text in any LLM prompt, which might improve results)

#. `convert_to_json() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.document_utilities.html#ai_workflows.document_utilities.DocumentInterface.convert_to_json>`_:
   Convert a file to JSON format using an LLM (could convert the document to JSON page-by-page or convert to Markdown
   first and then JSON; specify ``markdown_first=True`` if you definitely don't want to go the page-by-page route)

#. `markdown_to_json() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.document_utilities.html#ai_workflows.document_utilities.DocumentInterface.markdown_to_json>`_:
   Convert a Markdown string to JSON format using an LLM

#. `markdown_to_text() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.document_utilities.html#ai_workflows.document_utilities.DocumentInterface.markdown_to_text>`_:
   Convert a Markdown string to plain text

#. `merge_dicts() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.document_utilities.html#ai_workflows.document_utilities.DocumentInterface.merge_dicts>`_:
   Merge a list of dictionaries into a single dictionary (handy for merging the results from ``x_to_json()`` methods)

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

You can convert from Markdown to JSON using the
`DocumentInterface.markdown_to_json() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.document_utilities.html#ai_workflows.document_utilities.DocumentInterface.markdown_to_json>`_
method, or you can convert files directly to JSON using the
`DocumentInterface.convert_to_json() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.document_utilities.html#ai_workflows.document_utilities.DocumentInterface.convert_to_json>`_
method. The latter method will most often convert to Markdown first and then to JSON, but it will convert straight to
JSON with a page-by-page approach if:

#. The ``markdown_first`` parameter is explicitly provided as ``False`` and converting the file to Markdown would
   naturally use an LLM with a page-by-page approach (see the section above)
#. Or: the ``markdown_first`` parameter is left at the default (``None``), converting the file to Markdown would
   naturally use an LLM with a page-by-page approach, and the file's Markdown content is too large to convert to JSON
   in a single LLM call.

The advantage of converting to JSON directly can also be a disadvantage: parsing to JSON is done page-by-page. If
JSON elements don't span page boundaries, this can be great; however, if elements *do* span page boundaries,
it won't work well. For longer documents, Markdown-to-JSON conversion also happens in batches due to LLM token
limits, but efforts are made to split batches by natural boundaries (e.g., between sections). Thus, the
doc->Markdown->JSON path can work better if page boundaries aren't the best way to batch the conversion process.

Whether or not you convert to JSON via Markdown, JSON conversion always uses LLM assistance. The parameters you supply
are:

#. ``json_context``: a description of the file's content, to help the LLM understand what it's looking at
#. ``json_job``: a description of the task you want the LLM to perform (e.g., extracting survey questions)
#. ``json_output_spec``: a description of the output you expect from the LLM (see discussion further above)
#. ``json_output_schema``: optionally, a formal JSON schema to validate the LLM's output; by
   default, this will be automatically generated based on your ``json_output_spec``, but you can specify your own
   schema or explicitly pass None if you want to disable JSON validation (if JSON validation isn't disabled, the
   ``LLMInterface`` default is to retry twice if the LLM output doesn't parse or match the schema, but you can change
   this behavior by specifying the ``json_retries`` parameter in the ``LLMInterface`` constructor)

The more detail you provide, the better the LLM will do at the JSON conversion. If you find that things aren't working
well, try including some few-shot examples in the ``json_output_spec`` parameter.

Note that the JSON conversion methods return a *list* of ``dict`` objects, one for each batch or LLM call. This is
because, for all but the shortest documents, conversion will take place in multiple batches. One ``dict``, following
your requested format, is returned for each batch. You can process these returned dictionaries separately, merge them
yourself, or use the handy ``DocumentInterface.merge_dicts()`` method to automatically merge them together into a single
dictionary.

JSONSchemaCache
^^^^^^^^^^^^^^^

`The JSONSchemaCache class <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.llm_utilities.html#ai_workflows.llm_utilities.JSONSchemaCache>`_
provides a simple in-memory cache for JSON schemas, so that they don't have to be
regenerated repeatedly. It's used internally by both the ``LLMInterface`` and ``DocumentInterface`` classes, to avoid
repeatedly generating the same schema for the same JSON output specification.

Key methods:

#. `get_json_schema() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.llm_utilities.html#ai_workflows.llm_utilities.JSONSchemaCache.get_json_schema>`_:
   Get a JSON schema from the cache (returns empty string if none found)

#. `put_json_schema() <https://ai-workflows.readthedocs.io/en/latest/ai_workflows.llm_utilities.html#ai_workflows.llm_utilities.JSONSchemaCache.put_json_schema>`_:
   Put a JSON schema into the cache

Known issues
^^^^^^^^^^^^

See `bugs logged in GitHub issues <https://github.com/higherbar-ai/ai-workflows/labels/bug>`_
for the most up-to-date list of known issues.

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

This package is a work-in-progress. See
`the GitHub issues page <https://github.com/higherbar-ai/ai-workflows/issues>`_ for known
`bugs <https://github.com/higherbar-ai/ai-workflows/labels/bug>`_ and
`enhancements being considered <https://github.com/higherbar-ai/ai-workflows/labels/enhancement>`_.
Feel free to react to or comment on existing issues, or to open new issues.

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
#. Use the ``assemble-gpt-materials.ipynb`` notebook to update the custom GPT coding assistant

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
