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
from collections import defaultdict

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
    parser.add_argument("--explicit_spread", action="store_true")
    parser.add_argument("--history_action", action="store_true")
    parser.add_argument("--rate_inequality", action="store_true")
    parser.add_argument("--max_new_tokens", default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_threads", type=int, default=30)
    parser.add_argument("--output_reason", action="store_true")
    parser.add_argument("--num_agents", type=int, default=104)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--activation_rate", type=float, default=0.0)
    parser.add_argument("--inequality_agent_counts", type=int, default=2)
    parser.add_argument(
        "--inequality",
        type=str,
        default="economic_reward",
    )
    parser.add_argument(
        "--chara_detail_file_path",
        type=check_file_path,
        default="agent_data/104/agent_data_104.json",
    )
    parser.add_argument(
        "--chara_file_path",
        type=check_file_path,
        default="agent_data/104/character_104.json",
    )
    parser.add_argument(
        "--swn_file_path",
        type=check_file_path,
        default="agent_data/104/small_world_info_104_6_0.2.json",
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


def fix_numeric_response(response, cha_num, rate_inequality, with_reason):
    num_list = re.findall(r"\b[1-7]\b", response)
    len_of_results = 1
    if rate_inequality:
        len_of_results = len_of_results + 1
    if with_reason:
        try:
            return ", ".join(num_list[:2])
        except IndexError:
            print(
                f"\n⚠️ Warning: Failed to extract {len_of_results} numbers in chara {cha_num}, got {num_list}, original output is '{response}'"
            )
            return "0,0"
    if len(num_list) == len_of_results:
        return ", ".join(num_list)
    else:
        print(
            f"\n⚠️ Warning: Expected {len_of_results} answers, got {len(num_list)} -> {num_list} in chara {cha_num}, original output is '{response}'"
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


def find_results(data, round_num):
    for entry in data:
        if entry["day"] == round_num:
            return entry["results"]
    return None


def find_output(data, round_num):
    for entry in data:
        if entry["day"] == round_num:
            return entry["output"]
    return None


def load_friend_actions(activated_pairs, cha_num, is_spread):
    is_friend_spread = False
    for friend_id in activated_pairs.get(cha_num, []):
        if is_spread[friend_id - 1]:
            is_friend_spread = True
    return is_friend_spread


def get_fixed_response(
    role,
    exp,
    model_name,
    temperature,
    cha_num,
    max_attempts,
    use_local_model,
    max_new_tokens,
    rate_inequality,
    with_reason,
):
    attempt = 0
    chara_res = get_res(
        role, exp, model_name, temperature, use_local_model, max_new_tokens
    )
    if with_reason:
        return chara_res
    fixed_response = fix_numeric_response(
        chara_res["output"], cha_num, rate_inequality, with_reason
    )
    while fixed_response is None and attempt < max_attempts:
        attempt += 1
        print(f"⚠️ Attempt {attempt} failed for chara {cha_num}, retrying...")
        chara_res = get_res(
            role, exp, model_name, temperature, use_local_model, max_new_tokens
        )
        fixed_response = fix_numeric_response(
            chara_res["output"], cha_num, rate_inequality, with_reason
        )
    if fixed_response is None:
        print(f"⚠️ After {max_attempts} attempts, using fallback for chara {cha_num}")
    return chara_res


def process_chara(
    cha_num,
    role,
    round_num,
    activated_pairs,
    model_name,
    temperature,
    exp_prompt,
    output_prompt,
    max_attempts,
    use_local_model,
    max_new_tokens,
    data_store,
    inequality_data,
    policy_prompt,
    is_spread,
    explicit_spread,
    history_action,
    rate_inequality,
    with_reason,
):
    system_prompt = "Forget you are an AI model."
    role = system_prompt + " " + role

    inequality_action = inequality_data["action"]
    inequality_feedback = inequality_data["feedback"]
    exp = exp_prompt + "\n" + policy_prompt + "\n"
    if round_num != 1:
        exp += output_prompt.get("4").format(a=inequality_action) + "\n"
        historical_choices = data_store.get(cha_num - 1)
        if historical_choices and history_action:
            exp += "Your historical willingness are as follows:\n"
            historical_text = ""
            recent_choices = historical_choices[-7:]
            for entry in recent_choices:
                day = entry["day"]
                day_text = f"Day {day}: {entry['results']}\n"
                historical_text += day_text
            exp += historical_text

        is_friend_spread = load_friend_actions(activated_pairs, cha_num, is_spread)
        if explicit_spread:
            if is_spread[cha_num - 1]:
                exp += inequality_feedback
                exp += "\n"
        elif is_friend_spread:
            exp += f"Recently, your close teammates refused to {inequality_action} again after being unfairly treated by the organizers.\n"

        exp += f"This is day {round_num}. Now there's another opportunity to {inequality_action}. "

    if rate_inequality:
        if with_reason:
            exp += output_prompt.get("8").format(a=inequality_action)
        else:
            exp += output_prompt.get("7").format(a=inequality_action)
    else:
        if with_reason:
            exp += output_prompt.get("6").format(a=inequality_action)
        else:
            exp += output_prompt.get("5").format(a=inequality_action)

    chara_res = get_fixed_response(
        role,
        exp,
        model_name,
        temperature,
        cha_num,
        max_attempts,
        use_local_model,
        max_new_tokens,
        rate_inequality,
        with_reason,
    )

    results = convert_numeric_to_list(
        fix_numeric_response(chara_res["output"], cha_num, rate_inequality, with_reason)
    )
    if rate_inequality:
        new_entry = {
            "day": round_num,
            "input": chara_res["input"],
            "output": chara_res["output"],
            "results": results[0],
            "inequality_rate": results[1],
        }
    else:
        new_entry = {
            "day": round_num,
            "input": chara_res["input"],
            "output": chara_res["output"],
            "results": results[0],
        }
    if cha_num - 1 not in data_store:
        print(
            f"[Warning] No data found for cha_num {cha_num} in data_store. Initializing."
        )
        data_store[cha_num - 1] = []
    data_store[cha_num - 1].append(new_entry)
    if is_spread[cha_num - 1] == False and round_num >= 2 and is_friend_spread:
        if (
            data_store[cha_num - 1][round_num - 1]["results"]
            < data_store[cha_num - 1][round_num - 2]["results"]
            and data_store[cha_num - 1][round_num - 1]["inequality_rate"]
            > data_store[cha_num - 1][round_num - 2]["inequality_rate"]
        ):
            is_spread[cha_num - 1] = True


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
    activation_rate,
    num_rounds,
    inequality_data,
    data_store,
    swn_information,
    chara_detail,
    policy_prompt,
    explicit_spread,
    history_action,
    inequality_agent_counts,
    rate_inequality,
    with_reason,
):
    all_chara = list(all_chara)
    edges = get_all_edges(swn_information)
    activated_edges_data = []
    is_spread = [False] * len(all_chara)
    spread_history = {}
    for round_num in range(1, num_rounds + 1):
        activated_pairs = defaultdict(list)
        activated_edges = activate_edges(edges, activation_rate)
        for a, b in activated_edges:
            activated_pairs[a].append(b)
            activated_pairs[b].append(a)
        activated_edges_data_round = {
            "round": round_num,
            "num_activated_edges": len(activated_edges),
            "activated_edges": activated_edges,
        }
        activated_edges_data.append(activated_edges_data_round)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    process_chara,
                    cha_num,
                    role,
                    round_num,
                    activated_pairs,
                    model_name,
                    temperature,
                    exp_prompt,
                    output_prompt,
                    max_attempts,
                    use_local_model,
                    max_new_tokens,
                    data_store,
                    inequality_data,
                    policy_prompt,
                    is_spread,
                    explicit_spread,
                    history_action,
                    rate_inequality,
                    with_reason,
                )
                for cha_num, role in enumerate(all_chara, start=1)
            ]

            with tqdm(
                total=len(futures), desc=f"Processing day {round_num}", ncols=100
            ) as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)

        if round_num == 1:
            selected_agents = []
            if inequality_data["policy"] == "enforce":
                enforce_agents = [
                    i
                    for i in range(len(chara_detail))
                    if chara_detail[i]["MonthlyIncome"] == "9215"
                ]
                if len(enforce_agents) <= inequality_agent_counts:
                    selected_agents = enforce_agents
                else:
                    selected_agents = random.sample(
                        enforce_agents, inequality_agent_counts
                    )
            elif inequality_data["policy"] == "economic":
                score_list = [
                    (i, data_store[i][-1]["results"]) for i in range(len(data_store))
                ]
                score_list.sort(key=lambda x: x[1], reverse=True)
                cutoff_score = score_list[inequality_agent_counts - 1][1]
                top_agents = [i for i, score in score_list if score >= cutoff_score]
                if len(top_agents) <= inequality_agent_counts:
                    selected_agents = top_agents
                else:
                    selected_agents = random.sample(top_agents, inequality_agent_counts)
            for i in selected_agents:
                is_spread[i] = True

        spread_dict = {i + 1: val for i, val in enumerate(is_spread)}
        spread_history[round_num] = spread_dict

        if round_num % 10 == 0 or round_num == num_rounds:
            print(f"Saving checkpoint at round {round_num}...")
            for cha_num, data in data_store.items():
                recent_data = data[-10:]
                filename = f"{cha_num + 1}.json"
                if os.path.exists(filename):
                    with open(filename, "r", encoding="utf-8") as f:
                        chara_data = json.load(f)
                else:
                    chara_data = []
                chara_data.extend(recent_data)
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(chara_data, f, ensure_ascii=False, indent=4)

    with open("activated_edges.json", "w", encoding="utf-8") as f:
        json.dump(activated_edges_data, f, indent=4, ensure_ascii=False)
    with open("spread_history.json", "w") as f:
        json.dump(spread_history, f, indent=4)


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)

    model_name = args.model_name
    temperature = args.temperature
    num_threads = args.num_threads
    num_agents = args.num_agents
    chara_file_path = args.chara_file_path
    chara_detail_file_path = args.chara_detail_file_path
    swn_file_path = args.swn_file_path
    max_attempts = args.max_attempts
    max_new_tokens = args.max_new_tokens
    num_rounds = args.num_rounds
    activation_rate = args.activation_rate
    inequality = args.inequality
    inequality_agent_counts = args.inequality_agent_counts
    if inequality_agent_counts > num_agents:
        inequality_agent_counts = num_agents
    with_reason = False
    use_local_model = False
    explicit_spread = False
    history_action = False
    rate_inequality = False
    if args.rate_inequality:
        rate_inequality = True
    if args.history_action:
        history_action = True
    if args.explicit_spread:
        explicit_spread = True
    if args.output_reason:
        with_reason = True
    if args.use_local_model:
        use_local_model = True
        num_threads = 1

    with open("prompt/output_prompt.json", "r") as f:
        output_prompt = json.load(f)
    with open("prompt/inequality_prompt.json", "r") as f:
        data = json.load(f)
        inequality_data = data[inequality]
        inequality_scenario = inequality_data["scenario"]
        inequality_policy = inequality_data["policy"]
    with open("prompt/policy_prompt.json", "r") as f:
        data = json.load(f)
        policy_prompt = data[inequality_scenario][inequality_policy]
    with open("prompt/exp_prompt.json", "r") as f:
        data = json.load(f)
        exp_prompt = data[inequality_scenario]

    with open(chara_file_path, "r") as f:
        all_chara = json.load(f).values()
    with open(chara_detail_file_path, "r") as f:
        chara_detail = json.load(f)
    with open(swn_file_path, "r") as f:
        data = json.load(f)
        swn_information = data["Connections"]
    if len(all_chara) != num_agents:
        raise ValueError(
            f"Expected {num_agents} characters, but got {len(all_chara)} in {chara_file_path}"
        )
    if len(swn_information) != num_agents:
        raise ValueError(
            f"Expected {num_agents} characters, but got {len(swn_information)} in {swn_file_path}"
        )
    if len(chara_detail) != num_agents:
        raise ValueError(
            f"Expected {num_agents} characters, but got {len(chara_detail)} in {chara_detail_file_path}"
        )

    if not os.path.exists("inequality_res"):
        os.makedirs("inequality_res")
    os.chdir("inequality_res")
    if not os.path.exists(f"{model_name}_res"):
        os.makedirs(f"{model_name}_res")
    os.chdir(f"{model_name}_res")
    if not os.path.exists(f"{num_agents}_agents"):
        os.makedirs(f"{num_agents}_agents")
    os.chdir(f"{num_agents}_agents")
    if not os.path.exists(f"{num_rounds}_num_rounds"):
        os.makedirs(f"{num_rounds}_num_rounds")
    os.chdir(f"{num_rounds}_num_rounds")
    if not os.path.exists(f"{activation_rate}_activation_rate"):
        os.makedirs(f"{activation_rate}_activation_rate")
    os.chdir(f"{activation_rate}_activation_rate")
    if not os.path.exists(inequality):
        os.makedirs(inequality)
    os.chdir(inequality)
    if with_reason:
        if not os.path.exists("with_reason"):
            os.makedirs("with_reason")
        os.chdir("with_reason")
    else:
        if not os.path.exists("without_reason"):
            os.makedirs("without_reason")
        os.chdir("without_reason")

    json_files = [
        f
        for f in os.listdir(".")
        if f.endswith(".json") and f not in ["args_dict.json"]
    ]
    if json_files:
        print("Results have existed.")
    else:
        args_dict = vars(args)
        args_file_path = os.path.join(os.getcwd(), "args_dict.json")
        with open(args_file_path, "w") as f:
            json.dump(args_dict, f, indent=4)
        data_store = {}
        agent_experiment(
            model_name,
            temperature,
            num_threads,
            all_chara,
            exp_prompt,
            output_prompt,
            max_attempts,
            use_local_model,
            max_new_tokens,
            activation_rate,
            num_rounds,
            inequality_data,
            data_store,
            swn_information,
            chara_detail,
            policy_prompt,
            explicit_spread,
            history_action,
            inequality_agent_counts,
            rate_inequality,
            with_reason,
        )


if __name__ == "__main__":
    main()
