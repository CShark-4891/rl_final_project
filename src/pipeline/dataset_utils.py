"""Utilities to manage datasets, model saving, environments, etc."""


DATASETS = [
    # --- 1. Hopper Suite ---
    "mujoco/hopper/simple-v0",
    "mujoco/hopper/medium-v0",
    "mujoco/hopper/expert-v0",
    "mujoco/hopper/medium-replay-v0",
    # --- 2. Walker2d Suite ---
    "mujoco/walker2d/simple-v0",
    "mujoco/walker2d/medium-v0",
    "mujoco/walker2d/expert-v0",

    # --- 3. HalfCheetah Suite ---
    "mujoco/halfcheetah/simple-v0",
    "mujoco/halfcheetah/medium-v0",
    "mujoco/halfcheetah/expert-v0",

    # --- 4. Ant Suite ---
    #       "mujoco/ant/simple-v0",
    #       "mujoco/ant/medium-v0",
    "mujoco/ant/expert-v0",

    # --- 5. Humanoid Suite ---
    #      "mujoco/humanoid/simple-v0",
    #      "mujoco/humanoid/medium-v0",
    #      "mujoco/humanoid/expert-v0"
]

# Official D4RL reference baselines for normalized scoring calculations
# TODO: Needs Checking! => Find official reference from paper or repo
D4RL_REF_SCORES = {
    "halfcheetah-medium-v2": {"random": -280.05, "expert": 12135.0},
    "halfcheetah-random-v2": {"random": -280.05, "expert": 12135.0},
    "halfcheetah-expert-v2": {"random": -280.05, "expert": 12135.0},
    "halfcheetah-medium-replay-v2": {"random": -280.05, "expert": 12135.0},
    "halfcheetah-medium-expert-v2": {"random": -280.05, "expert": 12135.0},
    "hopper-medium-v2": {"random": -20.0, "expert": 3234.3},
    "hopper-random-v2": {"random": -20.0, "expert": 3234.3},
    "hopper-expert-v2": {"random": -20.0, "expert": 3234.3},
    "walker2d-medium-v2": {"random": 1.62, "expert": 4592.3},
    "walker2d-random-v2": {"random": 1.62, "expert": 4592.3},
    "walker2d-expert-v2": {"random": 1.62, "expert": 4592.3},
    "ant-expert-v2": {"random": -325.6, "expert": 3818.5},
    "hopper-medium-replay-v2": {"random": -20.0, "expert": 3234.3},
}

__DATASET_TO_GYM_ID = {
    "mujoco/hopper/simple-v0": "Hopper-v4",
    "mujoco/hopper/medium-v0": "Hopper-v4",
    "mujoco/hopper/expert-v0": "Hopper-v4",
    "mujoco/hopper/medium-replay-v0": "Hopper-v4",

    "mujoco/walker2d/simple-v0": "Walker2d-v4",
    "mujoco/walker2d/medium-v0": "Walker2d-v4",
    "mujoco/walker2d/expert-v0": "Walker2d-v4",

    "mujoco/halfcheetah/simple-v0": "HalfCheetah-v4",
    "mujoco/halfcheetah/medium-v0": "HalfCheetah-v4",
    "mujoco/halfcheetah/expert-v0": "HalfCheetah-v4",

    #       "mujoco/ant/simple-v0": "Ant-v4",
    #       "mujoco/ant/medium-v0": "Ant-v4",
    "mujoco/ant/expert-v0": "Ant-v4",

    #      "mujoco/humanoid/simple-v0": "Humanoid-v4",
    #      "mujoco/humanoid/medium-v0": "Humanoid-v4",
    #      "mujoco/humanoid/expert-v0": "Humanoid-v4"
}

__DATASET_TO_REF_ID = {
    "mujoco/hopper/simple-v0": "hopper-simple-v2",
    "mujoco/hopper/medium-v0": "hopper-medium-v2",
    "mujoco/hopper/expert-v0": "hopper-expert-v2",
    "mujoco/hopper/medium-replay-v0": "hopper-medium-replay-v2",

    "mujoco/walker2d/simple-v0": "walker2d-simple-v2",
    "mujoco/walker2d/medium-v0": "walker2d-medium-v2",
    "mujoco/walker2d/expert-v0": "walker2d-expert-v2",

    "mujoco/halfcheetah/simple-v0": "halfcheetah-simple-v2",
    "mujoco/halfcheetah/medium-v0": "halfcheetah-medium-v2",
    "mujoco/halfcheetah/expert-v0": "halfcheetah-expert-v2",

    #       "mujoco/ant/simple-v0": "ant-simple-v2",
    #       "mujoco/ant/medium-v0": "ant-medium-v2",
    "mujoco/ant/expert-v0": "ant-expert-v2",

    #      "mujoco/humanoid/simple-v0": "humanoid-simple-v2",
    #      "mujoco/humanoid/medium-v0": "humanoid-medium-v2",
    #      "mujoco/humanoid/expert-v0": "humanoid-expert-v2"
}

__DATASET_TO_PATH = {
    "mujoco/hopper/simple-v0": "hopper/simple",
    "mujoco/hopper/medium-v0": "hopper/medium",
    "mujoco/hopper/expert-v0": "hopper/expert",
    "mujoco/hopper/medium-replay-v0": "hopper/medium-replay",

    "mujoco/walker2d/simple-v0": "walker2d/simple",
    "mujoco/walker2d/medium-v0": "walker2d/medium",
    "mujoco/walker2d/expert-v0": "walker2d/expert",

    "mujoco/halfcheetah/simple-v0": "halfcheetah/simple",
    "mujoco/halfcheetah/medium-v0": "halfcheetah/medium",
    "mujoco/halfcheetah/expert-v0": "halfcheetah/expert",

    #       "mujoco/ant/simple-v0": "ant/simple",
    #       "mujoco/ant/medium-v0": "ant/medium",
    "mujoco/ant/expert-v0": "ant/expert",

    #      "mujoco/humanoid/simple-v0": "humanoid/simple",
    #      "mujoco/humanoid/medium-v0": "humanoid/medium",
    #      "mujoco/humanoid/expert-v0": "humanoid/expert"
}


def get_ref_score_from_dataset_name(dataset_str: str) -> float:
    """Returns the reference score for a given dataset string."""
    if dataset_str not in __DATASET_TO_REF_ID:
        raise ValueError(
            f"Dataset '{dataset_str}' is not recognized. Please check the dataset string.")
    ref_id = __DATASET_TO_REF_ID[dataset_str]
    if ref_id not in D4RL_REF_SCORES:
        raise ValueError(
            f"Reference scores for '{ref_id}' are not available in D4RL_REF_SCORES.")
    return D4RL_REF_SCORES[ref_id]


def get_gym_id_from_dataset_name(dataset_str: str) -> str:
    """Returns the Gymnasium ID for a given dataset string."""
    if dataset_str not in __DATASET_TO_GYM_ID:
        raise ValueError(
            f"Dataset '{dataset_str}' is not recognized. Please check the dataset string.")
    return __DATASET_TO_GYM_ID[dataset_str]


def get_path_from_dataset_name(dataset_str: str) -> str:
    """Returns the path for a given dataset string."""
    if dataset_str not in __DATASET_TO_PATH:
        raise ValueError(
            f"Dataset '{dataset_str}' is not recognized. Please check the dataset string.")
    return __DATASET_TO_PATH[dataset_str]
