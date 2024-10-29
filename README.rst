============
ai_workflows
============

This repository contains a toolkit for AI workflows (i.e., workflows that are pre-scripted and repeatable, but utilize
LLMs for various tasks). It's still in early development, but is ready to support piloting and experimentation.

Installation
------------

Install the latest version with pip::

    pip install py-ai-workflows

You'll also need to install several other dependencies, which you can do by running the ``initial-setup.ipynb`` Jupyter
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


Overview
---------

Here are the basics:

#. The ``llm_utilities`` module provides a simple interface for interacting with a large language model (LLM) via the
   `LangChain API <https://python.langchain.com/docs/>`_. It includes the ``LLMInterface`` class that can be used to
   interact with OpenAI's models in "JSON mode," so that you get structured responses parsed to dictionaries for
   programmatic use.

#. The ``document_utilities`` module provides an interface for extracting Markdown-formatted text from various file
   formats. It includes functions for reading Word, PDF, Excel, CSV, and HTML files, and then converting them into
   Markdown for use in LLM interactions. Simply create a ``DocumentInterface`` object and then call
   ``convert_to_markdown()`` to convert any file to Markdown. If you provide an ``LLMInterface`` object, the LLM will
   be used to help with high-quality conversion.
#. The ``example-google-colab-document-processing.ipynb`` notebook provides a simple Google Colab example of how to use
   the ``document_utilities`` module to convert files to Markdown format.
#. The ``example-testing.ipynb`` notebook provides a basic set-up for testing Markdown conversion methods (LLM-assisted
   vs. not-LLM-assisted).

Typical usage::

    from ai_workflows.llm_utilities import LLMInterface
    from ai_workflows.document_utilities import DocumentInterface

    llm_interface = LLMInterface(openai_api_key=openai_api_key)
    doc_interface = DocumentInterface(llm_interface=llm_interface)
    markdown = doc_interface.convert_to_markdown(file_path)

Technical notes
---------------

LLMInterface
^^^^^^^^^^^^

Currently, the ``LLMInterface`` class only works with OpenAI models, either directly from OpenAI or via Azure.

Claude support (directly or via AWS Bedrock) is next up on the roadmap, followed by other models as requested.

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

Roadmap
-------

There's much that can be improved here. For example:

* Unit testing
* Tracking and reporting LLM costs
* Improving evaluation and comparison methods
* Parallelizing LLM calls for faster processing
* Adding support for reading documents as JSON rather than Markdown (using an LLM parsing layer)
* Adding Claude and AWS Bedrock support
* Adding OCR support for PDF files when an LLM isn't available

Credits
-------

This toolkit was originally developed by `Higher Bar AI, PBC <https://higherbar.ai>`_, a public benefit corporation. To
contact us, email us at ``info@higherbar.ai``

Full documentation
------------------

See the full reference documentation here:

    https://ai-workflows.readthedocs.io/

Local development
-----------------

To develop locally:

#. ``git clone https://github.com/higherbar-ai/ai-workflows``
#. ``cd ai-workflows``
#. ``python -m venv venv``
#. ``source venv/bin/activate``
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
