import json
import os
from decimal import Decimal
from openai import OpenAI
import httpx
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils import get_model, predict, api_key
import re


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


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--use_local_model", action="store_true")
    parser.add_argument("--max_new_tokens", default=50)
    parser.add_argument("--num_threads", type=int, default=30)
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--output_reason", action="store_true")
    parser.add_argument("--num_rounds", type=int, default=60)
    return parser


TEMPERATURE = 1.0


with open(
    r"prompt/TPP_exp_prompt.json",
    "r",
) as f:
    exp_prompt = json.load(f)

with open(
    r"prompt/TPP_game_setting_prompt_54.json",
    "r",
) as f:
    game_setting = json.load(f)


def extract_game_setting(cha_num, round):
    for item in game_setting:
        if item["index"] == cha_num and item["trial"] == round + 1:
            amount_of_allocation = item["amount_of_allocation"]
            cost_level = item["cost_level"]
            amount_of_cost = item["amount_of_cost"]
            return {
                "amount_of_allocation": amount_of_allocation,
                "cost_level": cost_level,
                "amount_of_cost": amount_of_cost,
            }
    return None


with open(r"agent_data/104/character_104.json", "r") as json_file:
    all_chara = json.load(json_file).values()
all_chara = list(all_chara)


def get_api_res(role_message, exp_message, model_name):
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
            {"role": "system", "content": role_message},
            {"role": "user", "content": exp_message},
        ],
        temperature=TEMPERATURE,
        max_tokens=1500,
    )
    return response.choices[0].message.content


def get_local_res(role_message, exp_message, model_name, max_new_tokens):
    model_data = get_model_and_tokenizer(model_name, max_new_tokens, TEMPERATURE)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    generation_config = model_data["generation_config"]
    response = predict(model, tokenizer, generation_config, role_message, exp_message)
    return response


def get_res(
    role_message,
    exp_message,
    model_name,
    use_local_model,
    max_attempts,
    max_new_tokens,
):
    input_text = role_message + "\n" + exp_message
    # print(f"Input: {input_text}")
    default_res = {
        "AA_valence": 0,
        "AA_arousal": 0,
        "choice": 2,
        "AC_valence": 0,
        "AC_arousal": 0,
        "EmoFDBK_valence": 0,
        "EmoFDBK_arousal": 0,
        "input": input_text,
        "output": "",
    }
    attempt = 0
    output = None
    while attempt < max_attempts:
        try:
            if use_local_model:
                output = get_local_res(
                    role_message, exp_message, model_name, max_new_tokens
                )
            else:
                output = get_api_res(role_message, exp_message, model_name)
            # print(f"Attempt {attempt + 1} - Output: {output}")
            content = output.strip()
            match = re.search(
                r"(AA_valence\s*=\s*[-+]?\d*\.?\d+)"
                r"(,\s*AA_arousal\s*=\s*[-+]?\d*\.?\d+)?"
                r"(,\s*choice\s*=\s*\d+)?"
                r"(,\s*AC_valence\s*=\s*[-+]?\d*\.?\d+)?"
                r"(,\s*AC_arousal\s*=\s*[-+]?\d*\.?\d+)?",
                content,
                re.IGNORECASE,
            )
            if match:
                content = match.group(0)
            if content.endswith("."):
                content = content[:-1]
            res = default_res.copy()
            pairs = [p.strip() for p in content.split(",") if "=" in p]
            for pair in pairs:
                try:
                    key, value = [s.strip() for s in pair.split("=", 1)]
                    res[key] = Decimal(value)
                except (ValueError, IndexError):
                    continue
            res["EmoFDBK_valence"] = float(res["AC_valence"] - res["AA_valence"])
            res["EmoFDBK_arousal"] = float(res["AC_arousal"] - res["AA_arousal"])
            for key in ["AA_valence", "AA_arousal", "AC_valence", "AC_arousal"]:
                if key in res and isinstance(res[key], Decimal):
                    res[key] = float(res[key])
            for key in ["choice"]:
                if key in res and isinstance(res[key], Decimal):
                    res[key] = int(res[key])
            res["output"] = output
            return res

        except Exception as e:
            print(
                f"Attempt {attempt + 1} failed with error: {e}. Original output: {output}."
            )
            attempt += 1
            if attempt < max_attempts:
                print(f"Retrying... ({attempt}/{max_attempts})")
            else:
                print("Max attempts reached, returning default result.")
    default_res["output"] = output
    return default_res


def process_character(
    cha_num,
    num_rounds,
    model_name,
    use_local_model,
    max_attempts,
    max_new_tokens,
    with_reason,
):
    role = all_chara[cha_num]
    role_message = role + "\n" + exp_prompt["like_people"]
    character_res = []

    for round in tqdm(
        range(num_rounds), desc=f"Character {cha_num} rounds", leave=False
    ):
        x, level, y = (
            extract_game_setting(cha_num, round)["amount_of_allocation"],
            extract_game_setting(cha_num, round)["cost_level"],
            extract_game_setting(cha_num, round)["amount_of_cost"],
        )
        exp_message = exp_prompt["exp_rules"]
        exp_message += "\n" + exp_prompt["round_num"].format(a=round + 1)
        exp_message += " " + exp_prompt["round_setting"].format(
            a=x, b=30 - x, c=level, d=y
        )
        if with_reason:
            exp_message += "\n" + exp_prompt["with_reason"]
        else:
            exp_message += "\n" + exp_prompt["without_reason"]

        ont_res = {"cha_num": cha_num + 1, "round_num": round + 1}
        ont_res.update(
            get_res(
                role_message,
                exp_message,
                model_name,
                use_local_model,
                max_attempts,
                max_new_tokens,
            )
        )
        character_res.append(ont_res)

    return character_res


def gen_character_res(
    model_name,
    use_local_model,
    max_attempts,
    max_new_tokens,
    num_rounds,
    with_reason,
    num_threads,
):
    res = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(
                process_character,
                cha_num,
                num_rounds,
                model_name,
                use_local_model,
                max_attempts,
                max_new_tokens,
                with_reason,
            ): cha_num
            for cha_num in range(len(all_chara))
        }

        for future in tqdm(
            as_completed(futures), total=len(all_chara), desc="Processing characters"
        ):
            character_res = future.result()
            res.extend(character_res)

    res.sort(key=lambda x: (x["cha_num"], x["round_num"]))
    return res


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)

    model_name = args.model_name
    max_new_tokens = args.max_new_tokens
    num_threads = args.num_threads
    max_attempts = args.max_attempts
    num_rounds = args.num_rounds
    use_local_model = False
    with_reason = False
    if args.output_reason:
        with_reason = True
    if args.use_local_model:
        use_local_model = True
        num_threads = 1
    if not os.path.exists("TPP_res"):
        os.makedirs("TPP_res")
    os.chdir("TPP_res")
    if with_reason:
        json_filename = f"{model_name}_with_reason.json"
    else:
        json_filename = f"{model_name}.json"
    if os.path.exists(json_filename):
        print(f"{json_filename} has existed")
    else:
        args_dict = vars(args)
        if with_reason:
            args_file_path = os.path.join(
                os.getcwd(), f"{model_name}_with_reason_dict.json"
            )
        else:
            args_file_path = os.path.join(os.getcwd(), f"{model_name}_dict.json")
        with open(args_file_path, "w") as f:
            json.dump(args_dict, f, indent=4)
        res = gen_character_res(
            model_name,
            use_local_model,
            max_attempts,
            max_new_tokens,
            num_rounds,
            with_reason,
            num_threads,
        )
        with open(
            json_filename,
            "w",
        ) as json_file:
            json.dump(res, json_file)


if __name__ == "__main__":
    main()
