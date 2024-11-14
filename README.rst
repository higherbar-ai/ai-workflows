============
ai_workflows
============

The ``ai_workflows`` package is a toolkit for supporting AI workflows (i.e., workflows that are pre-scripted and
repeatable, but utilize LLMs for various tasks). It's still in early development, but is ready to support piloting and
experimentation.

Installation
------------

Install the latest version with pip::

    pip install py-ai-workflows

You'll also need to install several other dependencies, which you can do by running the
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

Finally, if you're accessing models via AWS Bedrock, the AWS CLI needs to be installed and configured for AWS access.

Jupyter notebooks with Google Colab support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use `the colab-or-not package <https://github.com/higherbar-ai/colab-or-not>`_ to initialize a Jupyter notebook
for Google Colab or other environments::

    %pip install colab-or-not

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

#. The ``llm_utilities`` module provides a simple interface for interacting with a large language model (LLM). It
   includes the ``LLMInterface`` class that can be used to interact with OpenAI's models in "JSON mode," so that you
   get structured responses parsed to dictionaries for programmatic use.
#. The ``document_utilities`` module provides an interface for extracting Markdown-formatted text from various file
   formats. It includes functions for reading Word, PDF, Excel, CSV, and HTML files, and then converting them into
   Markdown for use in LLM interactions. Simply create a ``DocumentInterface`` object and then call
   ``convert_to_markdown()`` to convert any file to Markdown. If you provide an ``LLMInterface`` object, the LLM will
   be used to help with high-quality conversion.
#. The `example-doc-conversion.ipynb <https://github.com/higherbar-ai/ai-workflows/blob/main/src/example-doc-conversion.ipynb>`_ notebook provides a simple example of how to use the ``document_utilities``
   module to convert files to Markdown format, in either Google Colab or a local environment.
#. The `example-surveyeval-lite.ipynb <https://github.com/higherbar-ai/ai-workflows/blob/main/src/example-surveyeval-lite.ipynb>`_
   notebook provides a more realistic workflow example that uses the ``document_utilities`` module to convert a survey
   file to Markdown format and then to JSON format, and then uses the ``llm_utilities`` module to evaluate survey
   questions using an LLM. It also works in either Google Colab or a local environment.
#. The `example-testing.ipynb <https://github.com/higherbar-ai/ai-workflows/blob/main/src/example-testing.ipynb>`_ notebook provides a basic set-up for testing Markdown conversion methods (LLM-assisted
   vs. not-LLM-assisted). At the moment, this notebook only works in a local environment.

Typical usage::

    from ai_workflows.llm_utilities import LLMInterface
    from ai_workflows.document_utilities import DocumentInterface

    llm_interface = LLMInterface(openai_api_key=openai_api_key)
    doc_interface = DocumentInterface(llm_interface=llm_interface)
    markdown = doc_interface.convert_to_markdown(file_path)
    dict_list = doc_interface.convert_to_json(
        file_path,
        json_context = "The file contains a survey instrument with questions to be administered to rural Zimbabwean household heads by a trained enumerator.",
        json_job = "Your job is to extract questions and response options from the survey instrument.",
        json_output_spec = "Return correctly-formatted JSON with the following fields: ..."
    )

Technical notes
---------------

LLMInterface
^^^^^^^^^^^^

When you use the ``LLMInterface`` class to interact with an LLM, you can provide a formal JSON schema to validate JSON
responses. If you only have a human-readable description of the JSON output format, you can convert that to a JSON
schema by calling ``LLMInterface.generate_json_schema()``. This method will convert a human-readable description into a
formal schema that can be used to validate JSON responses. (The ``DocumentInterface`` class will handle this
automatically when used to convert documents to JSON. See below for more details.)

By default, the ``LLMInterface`` class will retry twice if an LLM response doesn't parse as JSON or match the schema
provided (if any), but you can change this behavior by specifying the ``json_retries`` parameter in the constructor.

Currently, the ``LLMInterface`` class supports OpenAI models, either directly from OpenAI or via Azure, or the
Anthropic models, either directly or via AWS Bedrock.

Markdown conversion
^^^^^^^^^^^^^^^^^^^

The ``DocumentInterface.convert_to_markdown()`` method uses one of several methods to convert files to Markdown:

#. If an ``LLMInterface`` is available, PDF files are converted to Markdown with LLM assistance: we split the PDF into
   pages (splitting double-page spreads as needed), convert each page to an image, and then convert to Markdown using
   the help of a multimodal LLM. This is the most accurate method, but it's also the most expensive, running at about
   $0.015 per page as of October 2024. In the process, we try to keep narrative text that flows across pages together,
   drop page headers and footers, and describe images, charts, and figures as if to a blind person. We also do our best
   to convert tables to proper Markdown tables.
#. If an ``LLMInterface`` is not available, we use ``PyMuPDFLLM`` to convert PDF files to Markdown. This method
   doesn't handle images, charts, or figures, and it's pretty bad at tables, but it does a good job extracting text and
   a better job adding Markdown formatting than most other libraries. (``PyMuPDFLLM`` also supports a range of other
   file formats, and we also use it to convert them to Markdown. That includes ``.xps``, ``.epub``, ``.mobi``,
   ``.fb2``, ``.cbz``, ``.svg``, and ``.txt`` formats.)
#. For ``.xlsx`` files without charts or images, we use a custom parser to convert worksheets and table ranges to proper
   Markdown tables. If there are charts or images and we have an ``LLMInterface`` available, we use LibreOffice to
   convert to PDF and, if it's 10 pages or fewer, we convert from the PDF to Markdown using the LLM assistance method
   described above. If it's more than 10 pages, we fall back to the ``Unstructured`` method described below.
#. If we have an ``LLMInterface`` available, we use LibreOffice to convert ``.docx``, ``.doc``, and ``.pptx`` files to
   PDF and then convert the PDF to Markdown using the LLM assistance method described above. Otherwise, we fall back to
   the ``Unstructured`` method described below.
#. Finally, if we haven't managed to convert the file using one of the higher-quality methods described above, we use
   the ``Unstructured`` library to parse the file into elements and then add basic Markdown formatting. This method is
   fast and cheap, but it's also the least accurate.

JSON conversion
^^^^^^^^^^^^^^^

You can convert from Markdown to JSON using the ``DocumentInterface.markdown_to_json()`` method, or you can convert
files directly to JSON using the ``DocumentInterface.convert_to_json()`` method. The latter method will most often
convert to Markdown first and then to JSON, but it will convert straight to JSON with a page-by-page approach if:

#. The ``markdown_first`` parameter is explicitly provided as ``False`` and converting the file to Markdown would
   naturally use an LLM with a page-by-page approach (see the section above)
#. Or: converting the file to Markdown would naturally use an LLM with a page-by-page approach,
   the ``markdown_first`` parameter is not explicitly provided as ``True``, and the file's content doesn't look too
   large to fit in the LLM context window (<= 50 pages or 25,000 tokens).

The advantage of converting to JSON directly, bypassing the Markdown step, is that you can handle files of arbitrary
size. However, the page-by-page approach can work poorly for elements that span pages (since JSON conversion happens
page-by-page).

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
#. OpenAI can randomly refuse some requests to process PDF pages and currently there's no auto-retry, so it can
   interrupt PDF processing at inconvenient times. Azure's OpenAI implementation seems to be more robust, particularly
   if you relax the content filtering.
#. LangSmith tracing support is not as good for Anthropic as it is for OpenAI. Specifically: attached images don't show
   in traces and, while token usage does appear in the metadata, the UI doesn't show token usage nicely.

Roadmap
-------

There's much that can be improved here. For example:

* For what's already here:
    * Unit testing
    * Tracking and reporting LLM costs
    * Improving evaluation and comparison methods
    * Adding automatic retries for LLM refusals
    * Parallelizing LLM calls for faster processing
* Supporting more file formats and conversion methods:
    * Adding OCR support for PDF files when an LLM isn't available
    * Trying Claude's `direct PDF support <https://docs.anthropic.com/en/docs/build-with-claude/pdf-support>`_
    * Integrating IBM's `Docling <https://ds4sd.github.io/docling/>`_ and/or
      `Data Prep Kit <https://ibm.github.io/data-prep-kit/>`_ toolkits and benchmarking against other ingestion methods

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
