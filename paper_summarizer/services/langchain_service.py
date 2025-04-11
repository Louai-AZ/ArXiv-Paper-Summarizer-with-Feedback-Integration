import random
from typing import cast
from paper_summarizer.config import *
from langchain import hub
from langchainhub import Client as HubClient
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langsmith import Client

from paper_summarizer.services.huggingface_service import get_llm

hub_client = HubClient()
client = Client()

def _format_example(example):
    return f"""<example>
    <original>
    {example.inputs['input']}
    </original>
    <summary>
    {example.outputs['output']}
    </summary>
</example>"""

def few_shot_examples():
    if client.has_dataset(dataset_name=DATASET_NAME):
        examples = list(client.list_examples(dataset_name=DATASET_NAME))
        if not examples:
            return ""
        examples = random.sample(examples, min(len(examples), NUM_FEWSHOTS))
        return "\n".join([_format_example(e) for e in examples])
    return ""

def get_prompt_with_fewshots():
    few_shots = few_shot_examples()
    prompt = hub.pull(PROMPT_NAME)
    return prompt.partial(examples=few_shots)

def update_prompt_from_feedback(score, final_value):
    updated_prompts = [hub.pull(f"{PROMPT_NAME}:{c['commit_hash']}") for c in hub_client.list_commits(PROMPT_NAME)["commits"][:PROMPT_UPDATE_BATCHSIZE]]
    optimizer_prompt = hub.pull(OPTIMIZER_PROMPT_NAME)
    optimizer_llm = get_llm()
    conversation = f"<rating>User rated this {score}</rating>\n<turn idx=0>\nuser: feedback\n</turn idx=0>"

    system_template = optimizer_prompt | optimizer_llm | (lambda x: x.split("<improved_prompt>")[1].split("</improved_prompt>")[0].strip())
    updated_prompt = system_template.invoke({
        "prompt_versions": "\n\n".join([f"<prompt version={v}>\n{cast(SystemMessagePromptTemplate, p.messages[0]).prompt.template}\n</prompt>" for v, p in zip([c['commit_hash'] for c in hub_client.list_commits(PROMPT_NAME)["commits"][:PROMPT_UPDATE_BATCHSIZE]], updated_prompts)]),
        "current_prompt": cast(SystemMessagePromptTemplate, updated_prompts[0].messages[0]).prompt.template,
        "conversation": conversation,
        "final_value": final_value,
    })
    hub.push(PROMPT_NAME, ChatPromptTemplate.from_messages([("system", updated_prompt), MessagesPlaceholder(variable_name="messages")]))
