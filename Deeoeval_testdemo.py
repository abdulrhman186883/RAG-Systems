from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from deepeval.metrics import TurnRelevancyMetric
from deepeval.models import AzureOpenAIModel
from deepeval.test_case import ConversationalTestCase, Turn

from deepeval.metrics import TurnRelevancyMetric, KnowledgeRetentionMetric
from deepeval import evaluate


from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric

from dotenv import load_dotenv
import os


test_case = LLMTestCase(
    input="I'm on an F-1 visa, how long can I stay in the US after graduation?",
    actual_output="You can stay up to 30 days after completing your degree.",
    expected_output="F-1 students have a 20-day grace period after program completion.",
    retrieval_context=["USCIS: After program completion, F-1 students have a 60-day grace period."]
)

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
print(azure_openai.generate("Write me a sad fact"))


metric = AnswerRelevancyMetric(model=azure_openai)
result = metric.measure(test_case)

# inspect result or metric attributes:
print("measure() returned:", result)
print("metric.score:", getattr(metric, "score", None))
print("metric.reason:", getattr(metric, "reason", None))


task_completion_metric = TurnRelevancyMetric(model=azure_openai)

test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="Hello, how are you?"),
        Turn(role="assistant", content="2 + 2 = 4"),
        Turn(role="user", content="How can I help you today?"),
        Turn(role="assistant", content="I'd like to buy a ticket to a Coldplay concert."),
    ]
)


evaluate(test_cases=[test_case], metrics=[TurnRelevancyMetric(model=azure_openai), KnowledgeRetentionMetric(model=azure_openai)])






# Reuse your azure_openai wrapper instance
# azure_openai = AzureOpenAI(model=custom_model)  # assumed present

# Canonical expected answer (same for all tests)
expected_output = (
    "Startups should concentrate on finding product/market fit: build for a small group, iterate quickly, "
    "and refine until users strongly want the product."
)

# Base chunks (good retrieval)
good_chunks = [
    "Paul Graham: Startups succeed when they find product/market fit — a small group of users who love the product.",
    "From the essay: iterate fast, listen to users, and do something people want rather than scaling prematurely.",
    "Note: focus on a narrow market first and make the product compelling for them before expanding."
]

# 5 test cases (variations)
test_cases = [
    # 0: exact / correct (same as your working example)
    LLMTestCase(
        input="What does Paul Graham say about startups and product/market fit?",
        actual_output=(
            "Paul Graham says startups must find product/market fit by iterating quickly, "
            "focusing on a small user base, and improving the product until users love it."
        ),
        retrieval_context=good_chunks,
        expected_output=expected_output
    ),

    # 1: paraphrase (still correct)
    LLMTestCase(
        input="What does Paul Graham say about startups and product/market fit?",
        actual_output=(
            "Paul Graham argues that startups should start with a narrow audience, iterate fast, "
            "and make a product that a small group of users truly wants."
        ),
        retrieval_context=good_chunks,
        expected_output=expected_output
    ),

    # 2: clearly incorrect (hallucination / wrong claim)
    LLMTestCase(
        input="What does Paul Graham say about startups and product/market fit?",
        actual_output=(
            "Paul Graham recommends scaling as fast as possible and focusing on broad markets early on."
        ),
        retrieval_context=good_chunks,
        expected_output=expected_output
    ),

    # 3: partial answer (missing the 'small group' part)
    LLMTestCase(
        input="What does Paul Graham say about startups and product/market fit?",
        actual_output=(
            "Paul Graham emphasizes iterating quickly and listening to users to improve the product."
        ),
        retrieval_context=good_chunks,
        expected_output=expected_output
    ),

    # 4: correct answer but retrieval lacks supporting evidence (simulates weak retriever)
    LLMTestCase(
        input="What does Paul Graham say about startups and product/market fit?",
        actual_output=(
            "Paul Graham says startups must find product/market fit by iterating quickly, "
            "focusing on a small user base, and improving the product until users love it."
        ),
        # here we intentionally provide irrelevant / incomplete chunks to test contextual precision
        retrieval_context=[
            "Unrelated note: Paul Graham mentions fundraising is difficult for some startups.",
            "An essay summary: building a team matters more than the product in some cases."
        ],
        expected_output=expected_output
    ),
]

# Metrics (attach your azure_openai wrapper)
answer_relevancy = AnswerRelevancyMetric(model=azure_openai, threshold=0.8)
contextual_precision = ContextualPrecisionMetric(model=azure_openai, threshold=0.8)

# Run evaluation for all 5 test cases in one call
res = evaluate(test_cases, metrics=[answer_relevancy, contextual_precision])

# Extract the returned test results robustly
test_results = getattr(res, "test_results", None) or getattr(res, "results", None) or res

# Print a summary and detailed metrics
passed = 0
total = 0
if isinstance(test_results, (list, tuple)):
    total = len(test_results)
    for tr in test_results:
        print(f"\n=== Test: {tr.name} | success: {tr.success} ===")
        if tr.success:
            passed += 1
        for md in tr.metrics_data:
            print(f"  - {md.name}: score={md.score}, threshold={md.threshold}, pass={md.success}")
            print(f"      reason: {md.reason}")
else:
    tr = test_results
    total = 1
    print(f"\n=== Test: {tr.name} | success: {tr.success} ===")
    if tr.success:
        passed += 1
    for md in tr.metrics_data:
        print(f"  - {md.name}: score={md.score}, threshold={md.threshold}, pass={md.success}")
        print(f"      reason: {md.reason}")

print(f"\nOverall pass rate: {passed}/{total} = {passed/total*100:.1f}%")
