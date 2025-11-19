📊 Paul Graham RAG Evaluator — DeepEval Test Runner * paul_Deepeval.py *

This repository contains an enhanced DeepEval test runner designed to evaluate Retrieval-Augmented Generation (RAG) performance on a Paul Graham essay dataset.
It uses Azure OpenAI (GPT-5) as the judge model and runs multiple metrics (Answer Relevancy + Contextual Precision) on a batch of test cases loaded from CSV.

The script includes:

✔ Automatic loading of .env keys

✔ Custom DeepEval LLM wrapper for Azure OpenAI

✔ Automatic CSV input → test case conversion

✔ JSON / delimiter parsing for retrieval contexts

✔ Sanity-check failure test (ensures judge is not overly lenient)

✔ Full diagnostic printing for each metric

✔ Automatic CSV export of results

✔ Compact failure summary

✔ Warning if all tests pass (likely misconfiguration)



📚 Paul Graham RAG Chatbot * my_langChainPaulGhram.py *

A complete pipeline: scraping → parsing → chunking → embeddings → Chroma → RAG Q&A

This project builds a local RAG (Retrieval Augmented Generation) chatbot over all Paul Graham essays, using:

Unstructured.io HTML parser

LangChain Document abstraction

RecursiveCharacterTextSplitter` for chunking

Ollama for embeddings and LLM generation

ChromaDB for local vector retrieval

Similarity + score threshold search

It supports full offline/local inference when using Ollama models.

💻 Colab Notebook:
<[Colab Notebook](https://colab.research.google.com/gist/abdulrhman186883/e77a7373701bdcaaad96ef26b7f20844/semantic_chunking.ipynb)>

