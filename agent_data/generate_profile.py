import random
import numpy as np
import json
import os

data_distribution = {
    "Gender": {
        "type": "categorical",
        "values": ["Male", "Female"],
        "probabilities": [0.511, 0.489],
    },
    "AgeBracket": {
        "type": "categorical",
        "values": [
            "18-19",
            "20-24",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            "60-64",
            "65-69",
            "70-74",
            "75-79",
            "80-84",
            "85-89",
            "90-94",
            "95+",
        ],
        "probabilities": [
            0.043,
            0.061,
            0.070,
            0.094,
            0.098,
            0.084,
            0.085,
            0.105,
            0.100,
            0.070,
            0.068,
            0.054,
            0.033,
            0.020,
            0.011,
            0.004,
            0.001,
        ],
    },
    "MaritalStatus": {
        "type": "categorical",
        "values": ["Single", "Married", "Divorced", "Widowed"],
        "probabilities": [0.199, 0.712, 0.027, 0.063],
    },
    "MonthlyIncome": {
        "type": "categorical",
        "values": ["9215", "20442", "32195", "50220", "95055"],
        "probabilities": [0.2, 0.2, 0.2, 0.2, 0.2],
    },
    "EducationLevel": {
        "type": "categorical",
        "values": [
            "No Formal Education",
            "Primary",
            "JuniorHigh",
            "SeniorHigh",
            "CollegePlus",
        ],
        "probabilities": [0.04, 0.25, 0.36, 0.16, 0.19],
    },
    "Empathic_Concern": {"type": "normal", "mean": 3, "std": 0.5, "min": 1, "max": 5},
    "Moral_Identity": {"type": "normal", "mean": 3, "std": 0.5, "min": 1, "max": 5},
    "Social_Responsibility": {
        "type": "normal",
        "mean": 3,
        "std": 0.5,
        "min": 1,
        "max": 5,
    },
    "Altruistic_Tendency": {
        "type": "normal",
        "mean": 3,
        "std": 0.5,
        "min": 1,
        "max": 5,
    },
    "BigFive_Agreeableness": {
        "type": "normal",
        "mean": 3.69,
        "std": 0.47,
        "min": 1,
        "max": 5,
    },
    "BigFive_Conscientiousness": {
        "type": "normal",
        "mean": 3.29,
        "std": 0.59,
        "min": 1,
        "max": 5,
    },
    "BigFive_Neuroticism": {
        "type": "normal",
        "mean": 2.96,
        "std": 0.67,
        "min": 1,
        "max": 5,
    },
    "BigFive_Openness": {
        "type": "normal",
        "mean": 3.57,
        "std": 0.59,
        "min": 1,
        "max": 5,
    },
    "BigFive_Extraversion": {
        "type": "normal",
        "mean": 3.19,
        "std": 0.66,
        "min": 1,
        "max": 5,
    },
}


def pick_occupation_from_table():
    occupation_list = [
        ("Farmer", 0.004),
        ("Miner", 0.020),
        ("Worker", 0.219),
        ("MigrantWorker", 0.100),
        ("PowerTec", 0.022),
        ("Courier", 0.047),
        ("Programmer", 0.032),
        ("SelfEmp", 0.048),
        ("Waiter", 0.018),
        ("Trader", 0.042),
        ("HouseAgent", 0.031),
        ("Professor", 0.028),
        ("Consultant", 0.051),
        ("WaterEng", 0.016),
        ("CommunityStaff", 0.005),
        ("PrimaryTeacher", 0.119),
        ("Doctor", 0.069),
        ("Athlete", 0.009),
        ("CivilServant", 0.121),
    ]
    values = [o[0] for o in occupation_list]
    probs = [o[1] for o in occupation_list]
    return random.choices(values, weights=probs, k=1)[0]


def get_occupation(age):
    occupation = "unknown"
    if age < 21:
        occupation = "Student"
    elif age > 60:
        occupation = "Retired"
    else:
        unemp_prob = 0.052
        r = random.random()
        if r < unemp_prob:
            occupation = "Unemployed"
        else:
            occupation = pick_occupation_from_table()
    return occupation


def get_age_from_bracket(bracket):
    if bracket == "95+":
        return random.randint(95, 100)
    parts = bracket.split("-")
    return random.randint(int(parts[0]), int(parts[1]))


def generate_person(person_id):
    person = {}
    person["id"] = person_id
    for attr, meta in data_distribution.items():
        if meta["type"] == "categorical":
            person[attr] = random.choices(
                meta["values"], weights=meta["probabilities"], k=1
            )[0]
        elif meta["type"] == "normal":
            mean = meta["mean"]
            std = meta["std"]
            min_val = meta["min"]
            max_val = meta["max"]
            while True:
                sample = random.gauss(mu=mean, sigma=std)
                if sample >= min_val and sample <= max_val:
                    rounded = round(sample, 2)
                    # double-check boundary
                    if rounded < min_val:
                        rounded = min_val
                    if rounded > max_val:
                        rounded = max_val
                    person[attr] = rounded
                    break

    age = get_age_from_bracket(person["AgeBracket"])
    person["Age"] = age
    person["Occupation"] = get_occupation(age)

    return person


def generate_population_with_ids(num_samples):
    population = []
    for person_id in range(1, num_samples + 1):
        population.append(generate_person(person_id))
    return population


def save_population_to_file(population, filename):
    with open(filename, "w") as f:
        json.dump(population, f, indent=4, ensure_ascii=False)


num_agents = 5
folder_name = str(num_agents)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
os.chdir(folder_name)
file_name = f"agent_data_{num_agents}.json"
if os.path.exists(file_name):
    print(f"{file_name} has existed")
else:
    population_data = generate_population_with_ids(num_agents)
    save_population_to_file(population_data, file_name)
    print(f"Data has been saved to {file_name} file.")
