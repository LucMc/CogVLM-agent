import json
from typing import Any
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from pydantic import TypeAdapter

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize
from sat.model import AutoModel


from utils.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
from utils.models import CogAgentModel, CogVLMModel

from sat.mpu import get_model_parallel_world_size

parser = argparse.ArgumentParser()
parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
parser.add_argument("--chinese", action='store_true', help='Chinese interface')
parser.add_argument("--version", type=str, default="chat", choices=['chat', 'vqa', 'chat_old', 'base'], help='version of language process. if there is \"text_processor_version\" in model_config.json, thi>
parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')

parser.add_argument("--from_pretrained", type=str, default="cogagent-chat", help='pretrained ckpt')
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
parser.add_argument("--fp16", action="store_true", default=False)
parser.add_argument("--bf16", action="store_true", default=False)
parser.add_argument("--stream_chat", action="store_true")
args = parser.parse_args()
rank = int(os.environ.get('RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))
args = parser.parse_args()

model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=rank,
        rank=rank,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
        device='cpu' if args.quant else 'cuda',
        **vars(args)
    ), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
model = model.eval()

assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

language_processor_version = model_args.text_processor_version if 'text_processor_version' in model_args else args.version
print("[Language processor version]:", language_processor_version)
tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=language_processor_version)
image_processor = get_image_processor(model_args.eva_args["image_size"][0])
cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None
if args.quant:
    quantize(model, args.quant)
    if torch.cuda.is_available():
        model = model.cuda()
model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

def cog_decomposer(query, image_path='./Test_images/firefox.png', history=None, cache_image=None):
    try:
        pre_prompt = 'Explain the steps with grounding to '
        query = pre_prompt + query
        query = [query]
        image_path = [image_path]

        if world_size > 1:
            torch.distributed.broadcast_object_list(image_path, 0)
            torch.distributed.broadcast_object_list(query, 0)

        image_path = image_path[0]
        query = query[0]

        response, history, cache_image = chat(
            image_path,
            model,
            text_processor_infer,
            image_processor,
            query,
            history=history,
            cross_img_processor=cross_image_processor,
            image=cache_image,
            max_length=args.max_length,
            top_p=args.top_p,
            temperature=args.temperature,
            top_k=args.top_k,
            invalid_slices=text_processor_infer.invalid_slices,
            args=args
        )

        plan_start = response.find('Plan:')
        next_action_start = response.find('Next Action:')
        grounded_operation_start = response.find('Grounded Operation:')

        if plan_start != -1 and next_action_start != -1 and grounded_operation_start != -1:
            plan = response[plan_start + len('Plan:'):next_action_start].strip()
            next_action = response[next_action_start + len('Next Action:'):grounded_operation_start].strip()
            grounded_operation = response[grounded_operation_start + len('Grounded Operation:'):].strip()
        else:
            plan = ''
            next_action = ''
            grounded_operation = ''

        return plan, next_action, grounded_operation
    except Exception as e:
        print(e)
        return 'Error', '', ''


class MoEAgent():
    """An agent that uses GPT on the backend."""

    _FUNCTIONS: list[ChatCompletionToolParam] = [
        {"type": "function", "function": func}  # type: ignore
        for func in [
            {
                "name": "click",
                "description": "Click on an interactable element",
               # "parameters": TypeAdapter().json_schema(),
            },
            {
                "name": "type",
                "description": "Type into an input field",
                #"parameters": TypeAdapter().json_schema(),
            },
            {
                "name": "stop",
                "description": "Stops the decision making loop. Used when the task is finished.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]
    ]
    def __init__(self, model_name: str = "gpt-3.5-turbo-0125", temperature: float = 0.0, max_tokens: float=100.0):
        """Instantiates an agent using GPT on the backend.

        Args:
            model_name (str, optional): The name of the model to use. Defaults to "gpt-4-1106-preview".
            temperature (float, optional): The temperature to use for the model. Defaults to 0.0.
            presence_penalty (float, optional): Penalises repetition in the model. Defaults to 1.0.

        """
        # self._client=openai.OpenAI(
        #     base_url = "https://api.endpoints.anyscale.com/v1",
        #     api_key = "esecret_mnn7gzeilp9n3nkf3prn9jlsnq")
        # self._gpt_model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"
        self._client=openai.OpenAI(api_key="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")
        self._gpt_model_name='gpt-3.5-turbo-0125'
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens=max_tokens
        self.ACTING_USER_PROMPT = (
            "### TASK:\n{prompt}\n"
            "### PLAN:\n{plan}\n"
            "### NEXT ACTION:\n{next_action}\n"
            "### GROUNDED OPERATION:\n{grounded_operation}\n"
        )
        self.ACTING_SYSTEM_PROMPT=(
        "You are a web agent. You interact with web browsers to complete a user's given task. "
        "Use exclusively the provided information to respond with a function call for the next action to be taken towards completing the user's task. "
        "For the `click` and `type` actions, the `target` parameter must be a JSON object chosen from the interactable elements provided. "
        "You might find it useful the given Plan, Next Action and Grounded Operation")

    def act(self, prompt: str, plan: str, next_action: str, grounded_operation: str, previous_actions: list):
        previous_actions_str = "\n".join(previous_actions)
        content = self.ACTING_USER_PROMPT.format(
            prompt=prompt,
            plan=plan,
            next_action=next_action,
            grounded_operation=grounded_operation,
            previous_actions=previous_actions_str
        )

        completion = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {
                    "role": "system",
                    "content": self.ACTING_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            tools=self._FUNCTIONS,
            tool_choice="auto",
        )
        response = completion.choices[0].message.tool_calls
        assert response, "Model did not return a valid action."
        return response
   def run(self, initial_prompt: str, image_path: str):
        prompt = initial_prompt
        previous_actions = []

        while True:
            plan, next_action, grounded_operation = cog_decomposer(prompt, image_path)
            if plan == 'Error':
                print("An error occurred during cognitive decomposition.")
                break

            action = self.act(prompt, plan, next_action, grounded_operation, previous_actions)
            print("Action:", action)


## Change this condition to the correct way of identifying the 'stop' from the function calling. I just wanted it to work
            if str(action).find('stop')!=-1:
                print("Task completed.")
                break

            previous_actions.append(str(action))
            image_path=input('New SS:')

if __name__ == "__main__":
    agent = MoEAgent()

    initial_prompt = input("Enter the initial prompt: ")
    image_path = input("Enter the image path: ")
    agent.run(initial_prompt, image_path)

