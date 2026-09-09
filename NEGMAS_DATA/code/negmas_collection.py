# %%matplotlib inline
from random import choice
from negmas import (
    ResponseType,
    SAOMechanism,
    AspirationNegotiator,
    SAONegotiator,
    AdditiveFirstFollowingTBNegotiator,
    MappingUtilityFunction,
    LinearAdditiveUtilityFunction as LUFun,
    Issue,
    GeniusNegotiator,
    UtilityFunction,
)
from negmas.preferences.value_fun import LinearFun, IdentityFun, AffineFun
from negmas.inout import Scenario
from negmas.sao.negotiators import AspirationNegotiator
from negmas.genius.gnegotiators import BoulwareNegotiationParty
from negmas.genius import GeniusNegotiator
from negmas.genius.gnegotiators import AgreeableAgent2018, Agent33, ParsAgent, Caduceus, Atlas3, PonPokoAgent, MengWan, CaduceusDC16, YXAgent
import os
from collections import defaultdict
import itertools
import numpy as np
import csv


def _18_domain_tset(path):
    """
    Get absolute paths of all subfolders under the given root directory.
    :param path: root directory path
    :return: list of absolute domain folder paths
    """
    sub_folders = []
    domain_path = []
    # Check if the target path exists
    if os.path.exists(path):
        items = os.listdir(path)
        # Traverse all items and collect subdirectory names
        for item in items:
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                sub_folders.append(item)
    else:
        print("The specified path does not exist, please check the path!")
    # Generate absolute path for each subfolder
    for folder in sub_folders:
        absolute_path = os.path.join(path, folder)
        domain_path.append(absolute_path)
    return domain_path


def inverse_utility(ufun, utility, current_scenario, epsilon=0.01):
    """
    Find an offer whose utility is within [utility, utility + epsilon].
    :param ufun: utility function
    :param utility: target utility value
    :param current_scenario: negotiation scenario
    :param epsilon: tolerance threshold
    :return: matched offer, return the highest utility offer if no match found
    """
    sorted_utilities, _ = all_max_offer_and_offer_utility(current_scenario, ufun)
    for offer, current_utility in sorted_utilities:
        if current_utility >= utility and current_utility <= utility + epsilon:
            return offer
    return sorted_utilities[0][0]


def calculate_utility(ufun, offer, max_utilit):
    """
    Normalize utility value with small epsilon to avoid division by zero.
    :param ufun: utility function
    :param offer: negotiation offer
    :param max_utilit: maximum possible utility of this agent
    :return: normalized utility
    """
    if offer is None:
        return 0
    return (ufun(offer) + 1e-11) / (max_utilit + 1e-12)


def process_session_history(session_historys, oppo_agent_name):
    """
    Group negotiation traces by negotiation rounds.
    Fill incomplete round records for data consistency.
    :param session_historys: raw extended negotiation trace
    :param oppo_agent_name: opponent agent name
    :return: processed history grouped by rounds
    """
    session_historys_deal = []
    round_dict = {}
    for record in session_historys:
        round_num = record[0]
        if round_num not in round_dict:
            round_dict[round_num] = []
        round_dict[round_num].append(record)
    # Sort records by round number
    for round_num in sorted(round_dict.keys()):
        session_historys_deal.append(round_dict[round_num])

    # Supplement incomplete last round
    if len(session_historys_deal[-1]) < 2:
        if len(session_historys_deal) == 1:
            c_name = 'CaduceusDC16-33d164f3-223d-49c6-926a-7fcb21ad362e'
            new_c_name = c_name.replace('CaduceusDC16', oppo_agent_name)
            session_historys_deal[-1].append((session_historys_deal[-1][0][0], new_c_name, session_historys_deal[-1][0][2]))
        else:
            # Copy the last valid opponent offer to the final round if agreement is reached
            session_historys_deal[-1].append((session_historys_deal[-1][0][0], session_historys_deal[-2][1][1], session_historys_deal[-1][0][2]))
    return session_historys_deal


def run_negotiation(current_domain_path, my_Agent, my_Agent_name, oppo_Agent, oppo_Agent_name, max_rounds):
    """
    Launch one SAO negotiation session between two agents.
    :param current_domain_path: path to negotiation scenario domain
    :param my_Agent: agent class for our side
    :param my_Agent_name: name of our agent
    :param oppo_Agent: agent class for opponent
    :param oppo_Agent_name: opponent agent name
    :param max_rounds: maximum negotiation steps
    :return: session info, final agreement, negotiation trace, utility functions and scenario
    """
    scenario = Scenario.load(current_domain_path)
    my_Agent = my_Agent(preferences=scenario.ufuns[0], name=my_Agent_name)
    oppo_Agent = oppo_Agent(preferences=scenario.ufuns[1], name=oppo_Agent_name)
    session = scenario.make_session(n_steps=max_rounds)
    session.add(my_Agent)
    session.add(oppo_Agent)
    session_info = session.run()
    agreement = session.agreement
    session_historys = session.extended_trace
    my_Agent_ufuns = scenario.ufuns[0]
    my_Agent_preferences = my_Agent.preferences
    oppo_Agent_ufuns = scenario.ufuns[1]
    oppo_Agent_preferences = oppo_Agent.preferences
    return session_info, agreement, session_historys, my_Agent_ufuns, my_Agent_preferences, oppo_Agent_ufuns, oppo_Agent_preferences, scenario


def save_dataset_to_csv(dataset, my_Agent_name, save_folder):
    """
    Save MDP dataset to CSV file.
    Dataset keys: observations, actions, rewards, next_observations, terminals
    :param dataset: collected MDP dataset dict
    :param my_Agent_name: name of the main agent
    :param save_folder: folder to store csv file
    """
    data = defaultdict(list)
    new_filename = f"{my_Agent_name}_and_everagent_based_on_18domain.csv"
    full_path = os.path.join(save_folder, new_filename)

    if os.path.exists(full_path):
        print(f"File '{new_filename}' already exists, skip saving.")
        return

    # Flatten array and construct table rows
    for i in range(len(dataset["observations"])):
        observations = dataset["observations"][i].flatten().tolist()
        actions = [dataset["actions"][i]]
        rewards = [dataset["rewards"][i]]
        next_observations = dataset["next_observations"][i].flatten().tolist()
        terminals = [int(dataset["terminals"][i])]
        row = observations + actions + rewards + next_observations + terminals
        data["rows"].append(row)

    os.makedirs(save_folder, exist_ok=True)
    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = []
        header += [f"observations_{i}" for i in range(dataset["observations"].shape[1])]
        header += ["actions"]
        header += ["rewards"]
        header += [f"next_observations_{i}" for i in range(dataset["next_observations"].shape[1])]
        header += ["terminals"]
        writer.writerow(header)
        for row in data["rows"]:
            writer.writerow(row)


def load_dataset_from_csv(filename):
    """
    Load MDP dataset from saved CSV file and restore numpy arrays.
    Fixed hard-coded 6-dim observation space.
    :param filename: csv file path
    :return: dataset dictionary with numpy arrays
    """
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []

        for row in reader:
            row = list(map(float, row))
            obs_start = 0
            obs_end = 6
            obs = row[obs_start:obs_end]
            observations.append(obs)

            act_start = obs_end
            act_end = act_start + 1
            act = row[act_start:act_end][0]
            actions.append(act)

            rew_start = act_end
            rew_end = rew_start + 1
            rew = row[rew_start:rew_end][0]
            rewards.append(rew)

            next_obs_start = rew_end
            next_obs_end = next_obs_start + 6
            next_obs = row[next_obs_start:next_obs_end]
            next_observations.append(next_obs)

            term_start = next_obs_end
            term_end = term_start + 1
            term = bool(int(row[term_start:term_end][0]))
            terminals.append(term)

    dataset = {
        "observations": np.array(observations),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "next_observations": np.array(next_observations),
        "terminals": np.array(terminals)
    }
    return dataset


def make_dataset_on_18domain(domain_path, my_Agent, my_Agent_name, exist_aggent_list, max_rounds=5000):
    """
    Collect MDP transition samples by running negotiations across all domains and opponent agents.
    State: history of normalized utility values from recent rounds.
    Action: normalized utility of agent's current offer.
    Reward: utility difference between opponent offer and agent offer.
    Terminal: True if negotiation ends with agreement.
    :param domain_path: list of scenario domain paths
    :param my_Agent: main agent class
    :param my_Agent_name: main agent name
    :param exist_aggent_list: list of opponent agent classes and names
    :param max_rounds: maximum negotiation steps per session
    :return: assembled MDP dataset
    """
    dataset_state = []
    dataset_action = []
    dataset_reward = []
    dataset_next_state = []
    dataset_terminal = []

    for current_domain_path in domain_path:
        for opp_l in exist_aggent_list:
            oppo_Agent, oppo_Agent_name = opp_l[0], opp_l[1]
            if oppo_Agent_name == my_Agent_name:
                continue
            # Run negotiation session
            session_info, agreement, session_historys, my_Agent_ufuns, my_Agent_preferences, oppo_Agent_ufuns, oppo_Agent_preferences, current_scenario = run_negotiation(
                current_domain_path, my_Agent, my_Agent_name, oppo_Agent, oppo_Agent_name, max_rounds)
            _, my_Agent_max_utilit = all_max_offer_and_offer_utility(current_scenario, my_Agent_ufuns, my_Agent_preferences)
            _, oppo_Agent_max_utilit = all_max_offer_and_offer_utility(current_scenario, oppo_Agent_ufuns, oppo_Agent_preferences)
            session_historys_deal = process_session_history(session_historys, oppo_Agent_name)

            # Iterate each negotiation round to collect transitions
            for i in range(len(session_historys_deal)):
                current_round = session_historys_deal[i]
                next_round = session_historys_deal[i + 1] if i + 1 < len(session_historys_deal) else None

                # Build current state: 3 rounds * 2 values (opponent utility, agent utility) = 6 dims
                state = []
                for t in range(i - 2, i + 1):
                    if t < 0:
                        state.extend([0, 0])
                    else:
                        if len(session_historys_deal[t]) < 2:
                            state.extend([0, 0])
                        else:
                            opp_offer = session_historys_deal[t][1][2]
                            apn_offer = session_historys_deal[t][0][2]
                            opp_utility = calculate_utility(my_Agent_ufuns, opp_offer, my_Agent_max_utilit)
                            apn_utility = calculate_utility(my_Agent_ufuns, apn_offer, my_Agent_max_utilit)
                            state.extend([opp_utility, apn_utility])

                # Get current action and reward
                action = None
                current_offer = None
                current_oppo_offer = None
                reward = 0
                if len(current_round) >= 0:
                    current_offer = current_round[0][2]
                    current_oppo_offer = current_round[1][2]
                    action_utility = calculate_utility(my_Agent_ufuns, current_offer, my_Agent_max_utilit)
                    action = action_utility
                    reward = calculate_utility(my_Agent_ufuns, opp_offer, my_Agent_max_utilit) - calculate_utility(my_Agent_ufuns, apn_offer, my_Agent_max_utilit)

                # Build next state
                next_state = []
                for t in range(i + 1 - 2, i + 2):
                    if t < 0:
                        next_state.extend([0, 0])
                    else:
                        if t > len(session_historys_deal) - 1:
                            next_state.extend([0, 0])
                        else:
                            if len(session_historys_deal[t]) < 2:
                                next_state.extend([0, 0])
                            else:
                                opp_offer = session_historys_deal[t][1][2]
                                apn_offer = session_historys_deal[t][0][2]
                                opp_utility = calculate_utility(my_Agent_ufuns, opp_offer, my_Agent_max_utilit)
                                apn_utility = calculate_utility(my_Agent_ufuns, apn_offer, my_Agent_max_utilit)
                                next_state.extend([opp_utility, apn_utility])

                # Judge terminal flag
                terminal = False
                if agreement is not None and current_offer == agreement and (i + 1) == len(session_historys_deal):
                    terminal = True
                elif agreement is not None and current_oppo_offer == agreement and (i + 1) == len(session_historys_deal):
                    terminal = True

                dataset_state.append(state)
                dataset_action.append(action)
                dataset_reward.append(reward)
                dataset_next_state.append(next_state)
                dataset_terminal.append(terminal)

    # Assemble full dataset
    dataset = {}
    dataset["observations"] = np.array(dataset_state)
    dataset["actions"] = np.array(dataset_action)
    dataset["rewards"] = np.array(dataset_reward)
    dataset["next_observations"] = np.array(dataset_next_state)
    dataset["terminals"] = np.array(dataset_terminal)
    save_dataset_to_csv(dataset, my_Agent_name, r".\data_negotiation_V5")
    return dataset


def all_max_offer_and_offer_utility(current_scenario, ufun, current_agent_preferences):
    """
    Enumerate all possible issue combinations, compute utilities, sort in descending utility order.
    :param current_scenario: negotiation scenario
    :param ufun: agent utility function
    :param current_agent_preferences: agent preference model
    :return: sorted list of (offer, utility), maximum utility value
    """
    issues = current_agent_preferences.issues
    issue_values = [issue.values for issue in issues]
    all_combinations = list(itertools.product(*issue_values))
    utilities = []
    for combination in all_combinations:
        utility = ufun(combination)
        utilities.append((combination, utility))
    sorted_utilities = sorted(utilities, key=lambda x: x[1], reverse=True)
    max_u = sorted_utilities[0][1]
    return sorted_utilities, max_u


if __name__ == "__main__":
    # Agent pool definition
    exist_aggent_list = [
        [AgreeableAgent2018, 'AgreeableAgent2018'],
        [Atlas3, 'Atlas3'],
        [PonPokoAgent, 'PonPokoAgent'],
        [Caduceus, 'Caduceus'],
        [CaduceusDC16, 'CaduceusDC16'],
        [YXAgent, 'YXAgent'],
        [Caduceus, 'Caduceus'],
    ]
    # Load scenario folders
    domain_path = _18_domain_tset(path=r"F:\VSCODE_PYlearning\mygenius\压缩文件\negmas-master\tests\data\scenarios\anac\y2013")
    # Run data collection for each agent in the pool
    for i in range(len(exist_aggent_list)):
        dataset = make_dataset_on_18domain(
            domain_path,
            exist_aggent_list[i][0],
            exist_aggent_list[i][1],
            exist_aggent_list,
            max_rounds=500
        )
    # Load saved csv dataset for verification
    dataset1 = load_dataset_from_csv(r"F:\VSCODE_PYlearning\mygenius\negmas_learning\协商数据转MDP\data_negotiation_V5\AgreeableAgent2018_and_everagent_based_on_18domain.csv")
    print(dataset1['observations'][200])
    print("ac", dataset1['actions'][300])
    print("re", dataset1['rewards'][-1])
    print(dataset1['observations'][200])
    print(dataset1['next_observations'][200])
