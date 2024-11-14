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

from setuptools import setup, find_packages

with open('README.rst') as file:
    readme = file.read()


setup(
    name='py-ai-workflows',
    version='0.14.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.10',
    install_requires=[
        'unstructured[all-docs]',
        'openai',
        'anthropic[bedrock]',
        'tiktoken',
        'langsmith>=0.1.63,<0.2',
        'tenacity',
        'PyMuPDF',
        'pymupdf4llm',
        'Pillow',
        'openpyxl',
        'nltk==3.9.1',
        'jsonschema',
        'beautifulsoup4>=4.12.0',
        'markdown>=3.5.0',
    ],
    package_data={
        'ai_workflows': ['resources/*'], # include resource files in package
    },
    url='https://github.com/higherbar-ai/ai-workflows',
    project_urls={'Documentation': 'https://ai-workflows.readthedocs.io/'},
    license='Apache 2.0',
    author='Christopher Robert',
    author_email='crobert@higherbar.ai',
    description='A toolkit for AI workflows.',
    long_description=readme
)
