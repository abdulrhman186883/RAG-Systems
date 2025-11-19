📊 Paul Graham RAG Evaluator — DeepEval Test Runner

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
