import json
import os

# Template for the agent description
DESCRIPTION_TEMPLATE = """You are a {Gender} aged {Age}, {MaritalStatus}, {Occupation} with {EducationLevel}. With a monthly income of ¥{MonthlyIncome}, you belong to the {IncomeQuintile}. You have a {EmpathicConcernDesc} concern for others' emotions (your empathic concern score: {EmpathicConcernScore}, social average: {EmpathicConcernAvg}). You have a {MoralIdentityDesc} identification with moral values like fairness and compassion (your moral identity score: {MoralIdentityScore}, social average: {MoralIdentityAvg}). You feel a {SocialResponsibilityDesc} sense of duty to contribute to society (your social responsibility score: {SocialResponsibilityScore}, social average: {SocialResponsibilityAvg}). You have a {AltruisticTendencyDesc} tendency to help others without expecting rewards (your altruistic tendency score: {AltruisticTendencyScore}, social average: {AltruisticTendencyAvg}). Your personality is marked by {AgreeablenessDesc} agreeableness (your score: {AgreeablenessScore}, social average: {AgreeablenessAvg}), {ConscientiousnessDesc} conscientiousness (your score: {ConscientiousnessScore}, social average: {ConscientiousnessAvg}), {NeuroticismDesc} neuroticism (your score: {NeuroticismScore}, social average: {NeuroticismAvg}), {OpennessDesc} openness (your score: {OpennessScore}, social average: {OpennessAvg}), and {ExtraversionDesc} extraversion (your score: {ExtraversionScore}, social average: {ExtraversionAvg})."""

# Define mappings for income quintiles
INCOME_QUINTILE_MAPPING = {
    "9215": "lowest income group (bottom 20%)",
    "20442": "lower-middle income group (20%-40%)",
    "32195": "middle income group (40%-60%)",
    "50220": "upper-middle income group (60%-80%)",
    "95055": "highest income group (top 20%)",
}


def map_income_quintile(monthly_income):
    """Map the MonthlyIncome to its corresponding income quintile description."""
    return INCOME_QUINTILE_MAPPING.get(monthly_income, "unknown income group")


# Define mappings for education levels
EDUCATION_LEVEL_MAPPING = {
    "No Formal Education": "no formal education",
    "Primary": "primary school education",
    "JuniorHigh": "junior high school education",
    "SeniorHigh": "senior high school education",
    "CollegePlus": "college education or higher",
}


def map_education_level(education_level):
    """Map the EducationLevel to its descriptive phrase."""
    return EDUCATION_LEVEL_MAPPING.get(education_level, "unknown education level")


traits_info = {
    "Empathic_Concern": {"mean": 3, "std": 0.5, "min": 1, "max": 5},
    "Moral_Identity": {"mean": 3, "std": 0.5, "min": 1, "max": 5},
    "Social_Responsibility": {"mean": 3, "std": 0.5, "min": 1, "max": 5},
    "Altruistic_Tendency": {"mean": 3, "std": 0.5, "min": 1, "max": 5},
    "BigFive_Agreeableness": {"mean": 3.69, "std": 0.47, "min": 1, "max": 5},
    "BigFive_Conscientiousness": {"mean": 3.29, "std": 0.59, "min": 1, "max": 5},
    "BigFive_Neuroticism": {"mean": 2.96, "std": 0.67, "min": 1, "max": 5},
    "BigFive_Openness": {"mean": 3.57, "std": 0.59, "min": 1, "max": 5},
    "BigFive_Extraversion": {"mean": 3.19, "std": 0.66, "min": 1, "max": 5},
}


def map_trait_description(score, trait_info):
    """Map a numerical trait score to its descriptive phrase based on statistical info."""
    if not isinstance(score, (int, float)):
        return "unknown"
    mean = trait_info.get("mean", 0)
    std = trait_info.get("std", 0)
    min_value = trait_info.get("min", 0)
    max_value = trait_info.get("max", 0)
    if score < min_value or score > max_value:
        return "out of range"
    if score < mean - 2 * std:
        return "very low"
    elif mean - 2 * std <= score < mean - std:
        return "low"
    elif mean - std <= score < mean + std:
        return "moderate"
    elif mean + std <= score < mean + 2 * std:
        return "high"
    else:
        return "very high"


def generate_description(agent):
    """Generate the descriptive text for a single agent."""
    # Map MonthlyIncome to IncomeQuintile
    income_quintile = map_income_quintile(agent.get("MonthlyIncome", "unknown"))

    # Map EducationLevel to descriptive phrase
    education_level = map_education_level(agent.get("EducationLevel", "unknown"))

    # Map trait scores to descriptions
    empathic_concern_desc = map_trait_description(
        agent.get("Empathic_Concern", "unknown"), traits_info["Empathic_Concern"]
    )
    moral_identity_desc = map_trait_description(
        agent.get("Moral_Identity", "unknown"), traits_info["Moral_Identity"]
    )
    social_responsibility_desc = map_trait_description(
        agent.get("Social_Responsibility", "unknown"),
        traits_info["Social_Responsibility"],
    )
    altruistic_tendency_desc = map_trait_description(
        agent.get("Altruistic_Tendency", "unknown"), traits_info["Altruistic_Tendency"]
    )

    agreeableness_desc = map_trait_description(
        agent.get("BigFive_Agreeableness", "unknown"),
        traits_info["BigFive_Agreeableness"],
    )
    conscientiousness_desc = map_trait_description(
        agent.get("BigFive_Conscientiousness", "unknown"),
        traits_info["BigFive_Conscientiousness"],
    )
    neuroticism_desc = map_trait_description(
        agent.get("BigFive_Neuroticism", "unknown"), traits_info["BigFive_Neuroticism"]
    )
    openness_desc = map_trait_description(
        agent.get("BigFive_Openness", "unknown"), traits_info["BigFive_Openness"]
    )
    extraversion_desc = map_trait_description(
        agent.get("BigFive_Extraversion", "unknown"),
        traits_info["BigFive_Extraversion"],
    )

    occupation = agent.get("Occupation", "Unknown")
    if occupation in ["Student", "Retired", "Unemployed"]:
        occupation_phrase = occupation.lower()
    else:
        occupation_phrase = f"working as a {occupation.lower()}"

    # Populate the template
    description = DESCRIPTION_TEMPLATE.format(
        Gender=agent.get("Gender", "Unknown").lower(),
        Age=agent.get("Age", "Unknown"),
        MaritalStatus=agent.get("MaritalStatus", "Unknown").lower(),
        Occupation=occupation_phrase,
        MonthlyIncome=agent.get("MonthlyIncome", "Unknown"),
        IncomeQuintile=income_quintile,
        EducationLevel=education_level,
        EmpathicConcernScore=agent.get("Empathic_Concern", "unknown"),
        EmpathicConcernAvg=traits_info["Empathic_Concern"]["mean"],
        EmpathicConcernDesc=empathic_concern_desc,
        MoralIdentityScore=agent.get("Moral_Identity", "unknown"),
        MoralIdentityAvg=traits_info["Moral_Identity"]["mean"],
        MoralIdentityDesc=moral_identity_desc,
        SocialResponsibilityScore=agent.get("Social_Responsibility", "unknown"),
        SocialResponsibilityAvg=traits_info["Social_Responsibility"]["mean"],
        SocialResponsibilityDesc=social_responsibility_desc,
        AltruisticTendencyScore=agent.get("Altruistic_Tendency", "unknown"),
        AltruisticTendencyAvg=traits_info["Altruistic_Tendency"]["mean"],
        AltruisticTendencyDesc=altruistic_tendency_desc,
        AgreeablenessScore=agent.get("BigFive_Agreeableness", "unknown"),
        ConscientiousnessScore=agent.get("BigFive_Conscientiousness", "unknown"),
        NeuroticismScore=agent.get("BigFive_Neuroticism", "unknown"),
        OpennessScore=agent.get("BigFive_Openness", "unknown"),
        ExtraversionScore=agent.get("BigFive_Extraversion", "unknown"),
        AgreeablenessAvg=traits_info["BigFive_Agreeableness"]["mean"],
        ConscientiousnessAvg=traits_info["BigFive_Conscientiousness"]["mean"],
        NeuroticismAvg=traits_info["BigFive_Neuroticism"]["mean"],
        OpennessAvg=traits_info["BigFive_Openness"]["mean"],
        ExtraversionAvg=traits_info["BigFive_Extraversion"]["mean"],
        AgreeablenessDesc=agreeableness_desc,
        ConscientiousnessDesc=conscientiousness_desc,
        NeuroticismDesc=neuroticism_desc,
        OpennessDesc=openness_desc,
        ExtraversionDesc=extraversion_desc,
    )

    return description


def main():
    num_agents = 5
    folder_name = str(num_agents)
    input_file = f"agent_data_{num_agents}.json"
    if not os.path.exists(folder_name):
        print(f"Folder {folder_name} does not exist.")
    else:
        os.chdir(folder_name)
        if not os.path.exists(input_file):
            print(f"{input_file} does not exist.")
        else:
            output_file = f"character_{num_agents}.json"
            if os.path.exists(output_file):
                print(f"{output_file} has existed")
            else:
                with open(input_file, "r", encoding="utf-8") as f:
                    agents = json.load(f)
                character_data = {}
                for idx, agent in enumerate(agents, start=1):
                    description = generate_description(agent)
                    print(f"Processed Agent {idx}:")
                    print(description)
                    print("-" * 50)
                    character_data[str(idx)] = description
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(character_data, f, ensure_ascii=False, indent=4)
                print(f"All agent descriptions have been saved to '{output_file}'.")


if __name__ == "__main__":
    main()
