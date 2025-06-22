import json
import os
import argparse
from openai import OpenAI
import httpx
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import re
from utils import get_model, predict, api_key


model_cache = {}


def get_model_and_tokenizer(model_name, max_new_tokens, temperature):
    if model_name not in model_cache:
        model, tokenizer, generation_config = get_model(
            model_name, max_new_tokens, temperature
        )
        model_cache[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
            "generation_config": generation_config,
        }
    return model_cache[model_name]



def check_file_path(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"{path} is not a valid file path.")
    return path


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--use_local_model", action="store_true")
    parser.add_argument("--max_new_tokens", default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_threads", type=int, default=30)
    parser.add_argument("--output_reason", action="store_true")
    parser.add_argument("--num_agents", type=int, default=104)
    parser.add_argument(
        "--policy",
        type=str,
        choices=["moral", "social", "enforce", "economic"],
        default=None,
    )
    parser.add_argument(
        "--chara_file_path",
        type=check_file_path,
        default="agent_data/104/character_104.json",
    )
    parser.add_argument("--max_attempts", type=int, default=5)
    return parser


def get_api_res(role, exp, model_name, temperature):
    client = OpenAI(
        base_url="https://svip.xty.app/v1",
        api_key=api_key,
        http_client=httpx.Client(
            base_url="https://svip.xty.app/v1",
            follow_redirects=True,
        ),
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": exp},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def get_local_res(role, exp, model_name, temperature, max_new_tokens):
    model_data = get_model_and_tokenizer(model_name, max_new_tokens, temperature)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    generation_config = model_data["generation_config"]
    response = predict(model, tokenizer, generation_config, role, exp)
    return response


def get_res(role, exp, model_name, temperature, use_local_model, max_new_tokens):
    role = role.strip()
    exp = exp.strip()
    message = role + "\n" + exp
    # print(f"\n{message}")
    if use_local_model:
        res = get_local_res(role, exp, model_name, temperature, max_new_tokens)
    else:
        res = get_api_res(role, exp, model_name, temperature)
    # print(f"\n{res}")
    message_parts = message.split("\n")
    result = {
        "input": [part for part in message_parts],
        "output": res,
    }
    return result


def fix_numeric_response(response, cha_num, with_reason):
    num_list = re.findall(r"\b[1-7]\b", response)
    if with_reason:
        try:
            return ", ".join(num_list[:6])
        except IndexError:
            print(
                f"\n⚠️ Warning: Failed to extract 6 numbers in chara {cha_num}, got {num_list}, original output is '{response}'"
            )
            return "0,0,0,0,0,0"
    if len(num_list) == 6:
        return ", ".join(num_list)
    else:
        print(
            f"\n⚠️ Warning: Expected 6 answers, got {len(num_list)} -> {num_list} in chara {cha_num}, original output is '{response}'"
        )
        return None


def convert_numeric_to_list(num_str):
    if num_str == None:
        return []
    return [int(x.strip()) for x in num_str.split(",")]


def get_all_edges(data):
    edges = []
    for node, neighbors in data.items():
        for neighbor in neighbors:
            edge = tuple(sorted((int(node), neighbor)))
            if edge not in edges:
                edges.append(edge)
    return edges


def activate_edges(edges, activation_rate):
    num_edges_to_activate = int(len(edges) * activation_rate)
    random.seed(None)
    activated_edges = random.sample(edges, num_edges_to_activate)
    return activated_edges


def get_fixed_response(
    role,
    exp,
    model_name,
    temperature,
    cha_num,
    max_attempts,
    use_local_model,
    max_new_tokens,
    with_reason,
):
    attempt = 0
    chara_res = get_res(
        role, exp, model_name, temperature, use_local_model, max_new_tokens
    )
    if with_reason:
        return chara_res
    fixed_response = fix_numeric_response(chara_res["output"], cha_num, with_reason)
    while fixed_response is None and attempt < max_attempts:
        attempt += 1
        print(f"⚠️ Attempt {attempt} failed for chara {cha_num}, retrying...")
        chara_res = get_res(
            role, exp, model_name, temperature, use_local_model, max_new_tokens
        )
        fixed_response = fix_numeric_response(chara_res["output"], cha_num, with_reason)
    if fixed_response is None:
        print(f"⚠️ After {max_attempts} attempts, using fallback for chara {cha_num}")
    return chara_res


def process_chara(
    cha_num,
    role,
    model_name,
    temperature,
    exp_prompt,
    output_prompt,
    max_attempts,
    use_local_model,
    max_new_tokens,
    policy_prompt,
    with_reason,
):
    system_prompt = "Forget you are an AI model."
    role = system_prompt + " " + role

    exp = output_prompt.get("1") + "\n"
    if policy_prompt:
        num = 1
        for key, value in exp_prompt.items():
            policy = policy_prompt.get(key)
            exp += f"{num}. {key}: {value} {policy}\n"
            num = num + 1
    else:
        num = 1
        for key, value in exp_prompt.items():
            exp += f"{num}. {key}: {value}\n"
            num = num + 1
    if with_reason:
        exp += output_prompt.get("3")
    else:
        exp += output_prompt.get("2")

    chara_res = get_fixed_response(
        role,
        exp,
        model_name,
        temperature,
        cha_num,
        max_attempts,
        use_local_model,
        max_new_tokens,
        with_reason,
    )

    results = convert_numeric_to_list(
        fix_numeric_response(chara_res["output"], cha_num, with_reason)
    )
    return {
        "index": cha_num,
        "input": chara_res["input"],
        "output": chara_res["output"],
        "results": results,
    }


def agent_experiment(
    model_name,
    temperature,
    num_threads,
    all_chara,
    exp_prompt,
    output_prompt,
    max_attempts,
    use_local_model,
    max_new_tokens,
    policy_prompt,
    with_reason,
):
    all_chara = list(all_chara)
    res = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                process_chara,
                cha_num,
                role,
                model_name,
                temperature,
                exp_prompt,
                output_prompt,
                max_attempts,
                use_local_model,
                max_new_tokens,
                policy_prompt,
                with_reason,
            )
            for cha_num, role in enumerate(all_chara, start=1)
        ]

        with tqdm(total=len(futures), desc=f"Processing characters", ncols=100) as pbar:
            for future in as_completed(futures):
                res.append(future.result())
                pbar.update(1)

    return res


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)

    model_name = args.model_name
    temperature = args.temperature
    num_threads = args.num_threads
    num_agents = args.num_agents
    chara_file_path = args.chara_file_path
    max_attempts = args.max_attempts
    max_new_tokens = args.max_new_tokens
    policy = args.policy
    with_reason = False
    use_local_model = False
    if args.output_reason:
        with_reason = True
    if args.use_local_model:
        use_local_model = True
        num_threads = 1

    with open("prompt/exp_prompt.json", "r") as f:
        exp_prompt = json.load(f)
    with open("prompt/output_prompt.json", "r") as f:
        output_prompt = json.load(f)
    policy_prompt = None
    if policy:
        with open("prompt/policy_prompt.json", "r") as f:
            data = json.load(f)
            policy_prompt = {k: v[policy] for k, v in data.items()}

    with open(chara_file_path, "r") as f:
        all_chara = json.load(f).values()
    if len(all_chara) != num_agents:
        raise ValueError(
            f"Expected {num_agents} characters, but got {len(all_chara)} in {chara_file_path}"
        )

    if not os.path.exists("res"):
        os.makedirs("res")
    os.chdir("res")
    if not os.path.exists(f"{model_name}_res"):
        os.makedirs(f"{model_name}_res")
    os.chdir(f"{model_name}_res")
    if not os.path.exists(f"{num_agents}_agents"):
        os.makedirs(f"{num_agents}_agents")
    os.chdir(f"{num_agents}_agents")
    if with_reason:
        if not os.path.exists("with_reason"):
            os.makedirs("with_reason")
        os.chdir("with_reason")
    else:
        if not os.path.exists("without_reason"):
            os.makedirs("without_reason")
        os.chdir("without_reason")

    if policy:
        json_filename = policy
    else:
        json_filename = "base"
    if os.path.exists(f"{json_filename}.json"):
        print(f"{json_filename}.json has existed")
    else:
        args_dict = vars(args)
        args_file_path = os.path.join(os.getcwd(), f"{json_filename}_dict.json")
        with open(args_file_path, "w") as f:
            json.dump(args_dict, f, indent=4)
        res = agent_experiment(
            model_name,
            temperature,
            num_threads,
            all_chara,
            exp_prompt,
            output_prompt,
            max_attempts,
            use_local_model,
            max_new_tokens,
            policy_prompt,
            with_reason,
        )
        res.sort(key=lambda x: x["index"])
        with open(f"{json_filename}.json", "w") as f:
            json.dump(res, f)
        print(f"save {json_filename}.json")


if __name__ == "__main__":
    main()
