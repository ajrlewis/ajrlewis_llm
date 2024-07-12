"""
https://python.langchain.com/v0.2/docs/integrations/llms/huggingface_pipelines/
"""
import os
from langchain_core.prompts.prompt import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)
from loguru import logger

HF_API_KEY = os.getenv("HF_API_KEY")
logger.debug(f"{HF_API_KEY = }")

question_template = "Question: {question}. Return only the concise answer. Answer:"
summarize_template = "Summarize fully in your own words the following text: {text}"


def call_llm(
    inputs: dict,
    template: str,
    model_id="google/flan-t5-xl",
    model_task="text2text-generation",
    model_config={
        # "max_length": 60,
        "max_new_tokens": 1000,
        "temperature": 0.2,
        "do_sample": True
        # "top_k": 1,
    },
) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_API_KEY)
    # logger.debug(f"{tokenizer = }")
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=HF_API_KEY)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=HF_API_KEY)
    # logger.debug(f"{model = }")
    pipe = pipeline(model_task, model=model, tokenizer=tokenizer, **model_config)
    # logger.debug(f"{pipe = }")
    llm = HuggingFacePipeline(pipeline=pipe)
    # logger.debug(f"{llm = }")
    prompt = PromptTemplate(template=template, input_variables=list(inputs.keys()))
    logger.debug(f"{prompt = }")
    chain = prompt | llm
    logger.debug(f"{chain = }")
    output = chain.invoke(inputs)
    logger.debug(f"{output = }")
    # outputs = chain.batch(inputs)
    # for output in outputs:
    #     logger.debug(f"{output = }")
    return output


def main():
    inputs = {
        "text": 'Oh, so now even Elon admits that politics in a "democracy" is just a theater? But puppet masters are real. And Satoshi is one of the factions among them.'
    }
    output = call_llm(
        inputs,
        summarize_template,
        model_task="text-generation",
        # model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    )
    logger.debug(f"{output = }")


if __name__ == "__main__":
    main()


# text = "Find me 5 leads similar to phippsint.com located in US in the hotel fixtures and fitting manufacturing industry. Extract contacts with the following job titles project manager, head of purchasing, CEO."
# inputs = [
#     {
#         "question": f"Extract the website URL of the reference company from this text: `{text}`"
#     },
#     {"question": f"Extract the desired number of new leads from this text: `{text}`"},
#     {"question": f"Extract the desired location of new leads from this text: `{text}`"},
#     {
#         "question": f"Extract any additional keyword for the desired new leads from this text: `{text}`"
#     },
#     {
#         "question": f"Extract any desired job titles of employees in the new leads from this text: `{text}`"
#     },
# ]
