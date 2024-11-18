import numpy as np

from mushroom_rl.utils.dataset import compute_J, compute_metrics

def compute_SR(dataset, dataset_info = None):
    success_rate = list()

    if dataset_info:
        success = False
        episode_steps = 0
        for i, (d_i, sr_i) in enumerate(zip(dataset, dataset_info["success"])):
            success |= bool(int(sr_i)) # TODO: refine that part
            episode_steps += 1
            if d_i[-1] or i == len(dataset) - 1:
                success_rate.append(int(success))
                success = False
                episode_steps = 0
    else:
        for i in range(len(dataset)):
            if dataset[i][-1]:
                success_rate.append(1.0 if dataset[i][-2] else 0.)

    if len(success_rate) == 0:
        return [0]
    return success_rate

def get_stats(dataset, gamma, gamma_eval, dataset_info = None):
    min_J, max_J, mean_J, _, _ = compute_metrics(dataset, gamma_eval)
    mean_discounted_J = np.mean(compute_J(dataset, gamma))
    success_rate = np.mean(compute_SR(dataset, dataset_info=dataset_info))

    return min_J, max_J, mean_J, mean_discounted_J, success_rate

def parse_dataset(dataset, features=None, n_contexts = 0):
    """
    Split the dataset in its different components and return them.

    Args:
        dataset (list): the dataset to parse;
        features (object, None): features to apply to the states.

    Returns:
        The np.ndarray of contexts, state, action, reward, next_state, absorbing flag and
        last step flag. Features are applied to ``state`` and ``next_state``,
        when provided.

    """
    assert len(dataset) > 0

    shape = dataset[0][0][1].shape if features is None else (features.size,)
    
    contexts = np.ones(len(dataset), dtype=np.int64)
    state = np.ones((len(dataset),) + shape)
    action = np.ones((len(dataset),) + dataset[0][1].shape)
    reward = np.ones(len(dataset))
    next_state = np.ones((len(dataset),) + shape)
    absorbing = np.ones(len(dataset))
    last = np.ones(len(dataset))

    if features is not None:
        for i in range(len(dataset)):
            contexts[i] = dataset[i][0][0]
            state[i, ...] = features(dataset[i][0][1])
            action[i, ...] = dataset[i][1]
            reward[i] = dataset[i][2]
            next_state[i, ...] = features(dataset[i][3][1])
            absorbing[i] = dataset[i][4]
            last[i] = dataset[i][5]
    else:
        for i in range(len(dataset)):
            contexts[i] = dataset[i][0][0]
            state[i, ...] = dataset[i][0][1]
            action[i, ...] = dataset[i][1]
            reward[i] = dataset[i][2]
            next_state[i, ...] = dataset[i][3][1]
            absorbing[i] = dataset[i][4]
            last[i] = dataset[i][5]

    if n_contexts > 0:
        return np.array(contexts), np.array(state), np.array(action), np.array(reward), np.array(
            next_state), np.array(absorbing), np.array(last)

    return np.array(state), np.array(action), np.array(reward), np.array(
        next_state), np.array(absorbing), np.array(last)
