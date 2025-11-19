# paul_test.py — enhanced with diagnostic printing and sanity control test
# Original: user-provided test runner for DeepEval metrics
# Additions: detailed per-test metric printing, an obvious-failure control test,
# and a final sanity warning if everything passes.

import csv
import json
import traceback
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric
from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv
import os





class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"
    

load_dotenv()  


Azure_endpoint = os.getenv("Azure_endpoint")
Openai_api_key = os.getenv("key")


print(Azure_endpoint,Openai_api_key)


custom_model = AzureChatOpenAI(
    openai_api_version="2024-08-01-preview",
    azure_deployment="gpt-5",
    azure_endpoint=Azure_endpoint,
    openai_api_key=Openai_api_key,
)
azure_openai = AzureOpenAI(model=custom_model)


INPUT_CSV = "paul_graham_tests.csv"
OUTPUT_CSV = "paul_graham_results.csv"


def parse_retrieval_context(field_text: str):
    """
    Accepts either:
      - a JSON array string: '["a", "b"]'
      - OR a separator-based string: 'chunk1 ||| chunk2 ||| chunk3'
    Returns a list[str].
    """
    if field_text is None:
        return []
    s = field_text.strip()
    if not s:
        return []
    # try JSON parse first
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    # fallback: split by |||
    if "|||" in s:
        parts = [p.strip() for p in s.split("|||") if p.strip()]
        return parts
    # fallback: treat whole field as single chunk
    return [s]


def read_test_cases_from_csv(path):
    tests = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            tid = row.get("id") or f"test_{i}"
            input_text = row.get("input", "") or ""
            actual_output = row.get("actual_output", "") or ""
            expected_output = row.get("expected_output", "")
            retrieval_context = parse_retrieval_context(row.get("retrieval_context", "") or "")
            notes = row.get("notes", "") or ""
            # Build LLMTestCase (expected_output can be None but some metrics require it)
            tc = LLMTestCase(
                input=input_text,
                actual_output=actual_output,
                expected_output=expected_output,
                retrieval_context=retrieval_context
            )
            # attach id and notes as metadata (not required, but convenient)
            tc._id = tid
            tc._notes = notes
            tests.append(tc)
    return tests


def extract_test_results(evaluate_return):
    # Handle a few deepeval return shapes
    tr = getattr(evaluate_return, "test_results", None) or getattr(evaluate_return, "results", None) or evaluate_return
    if isinstance(tr, (list, tuple)):
        return tr
    return [tr]


def save_results_csv(test_results, path_out):
    # Build header dynamically: base cols + per metric (score/reason/success)
    # Collect all metric names to create columns
    metric_names = set()
    for tr in test_results:
        for md in tr.metrics_data:
            metric_names.add(md.name)
    metric_names = sorted(metric_names)

    # header
    base_cols = ["id", "name", "success", "input", "actual_output", "expected_output", "retrieval_context", "notes"]
    metric_cols = []
    for m in metric_names:
        metric_cols += [f"{m}__score", f"{m}__reason", f"{m}__success"]
    header = base_cols + metric_cols

    with open(path_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for tr in test_results:
            row = {c: "" for c in header}
            row["id"] = getattr(tr, "name", "") or getattr(tr, "_id", "")
            row["name"] = getattr(tr, "name", "")
            row["success"] = getattr(tr, "success", "")
            row["input"] = getattr(tr, "input", "") or ""
            row["actual_output"] = getattr(tr, "actual_output", "") or ""
            row["expected_output"] = getattr(tr, "expected_output", "") or ""
            row["retrieval_context"] = "|~|".join(getattr(tr, "retrieval_context", []) or [])
            row["notes"] = getattr(tr, "additional_metadata", "") or ""
            for md in tr.metrics_data:
                base = md.name
                row[f"{base}__score"] = md.score
                row[f"{base}__reason"] = md.reason
                row[f"{base}__success"] = md.success
            writer.writerow(row)
    print(f"Saved results to {path_out}")


def main():
    try:
        test_cases = read_test_cases_from_csv(INPUT_CSV)
        print(f"Loaded {len(test_cases)} test cases from {INPUT_CSV}")
        if not test_cases:
            print("No test cases found; exiting.")
            return

        
        metrics = [
            AnswerRelevancyMetric(model=azure_openai, threshold=0.95),
            ContextualPrecisionMetric(model=azure_openai, threshold=0.95),
        ]

        # testing the model 
        # --- ADD: sanity control test (should fail) ---
        nonsense = LLMTestCase(
            input="What is product/market fit?",
            actual_output="Banana.",
            expected_output="Startups should build for a small group, iterate quickly, and refine until users want the product.",
            retrieval_context=["irrelevant chunk"]
        )
        nonsense._id = "sanity_nonsense"
        test_cases.append(nonsense)
        print("Appended sanity control test (sanity_nonsense).")

        # Run evaluation (single call for all test_cases)
        print("Running evaluate() on batch...")
        res = evaluate(test_cases, metrics=metrics)
        print("evaluate() returned:", type(res), "\n")

        # === DIAGNOSTIC: pretty-print all metric details to console ===
        test_results = extract_test_results(res)
        print(f"Got {len(test_results)} test_results. Detailed metrics follow:\n")
        for tr in test_results:
            tid = getattr(tr, "name", "") or getattr(tr, "_id", "")
            print(f"== Test: {tid}  (success={getattr(tr,'success',None)}) ==")
            print(" Input:", getattr(tr, "input", "")[:200])
            print(" Actual:", getattr(tr, "actual_output", "")[:300])
            print(" Expected:", getattr(tr, "expected_output", "")[:300])
            print(" Retrieval chunks:", getattr(tr, "retrieval_context", []))
            for md in tr.metrics_data:
                print(f"  - Metric: {md.name}")
                print(f"      score : {md.score}")
                print(f"      success : {md.success}")
                # reason can be long; print first 800 chars
                reason = md.reason if md.reason is not None else ""
                print(f"      reason (truncated): {str(reason)[:800]}")
            print("\n")

        # Save CSV results as before
        save_results_csv(test_results, OUTPUT_CSV)

        # Print compact summary of failing tests (if any)
        fails = [tr for tr in test_results if not getattr(tr, "success", False)]
        if fails:
            print(f"\nFailures: {len(fails)} / {len(test_results)}")
            for tr in fails:
                name = getattr(tr, "name", "") or getattr(tr, "_id", "")
                print(f"- {name}: success={tr.success}")
                for md in tr.metrics_data:
                    if not md.success:
                        print(f"    * {md.name} -> score={md.score} reason={md.reason}")
        else:
            print("\nAll tests passed!")

        # warn user if everything passed (100%), 
        total = len(test_results)
        passed = sum(1 for tr in test_results if getattr(tr, "success", False))
        if total > 0 and passed == total:
            print("\nWARNING: All tests passed (100%). This can be realistic but often indicates a lenient judge or misconfiguration.")
            # show a compact table of metric scores for inspection
            for tr in test_results:
                tid = getattr(tr, "name", "") or getattr(tr, "_id", "")
                print(f"{tid}: success={getattr(tr,'success',False)}")
                for md in tr.metrics_data:
                    print(f"  {md.name} -> score={md.score} success={md.success}")

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
