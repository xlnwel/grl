class EnvSpec:
    """A specification for a particular instance of the environment. Used
    to register the parameters for official evaluations.

    Args:
        id (str): The official environment ID
        entry_point (Optional[str]): The Python entrypoint of the environment class (e.g. module.name:Class)
        reward_threshold (Optional[int]): The reward threshold before the task is considered solved
        nondeterministic (bool): Whether this environment is non-deterministic even after seeding
        max_episode_steps (Optional[int]): The maximum number of steps that an episode can consist of
        kwargs (dict): The kwargs to pass to the environment class

    """

    def __init__(self, id, entry_point=None, reward_threshold=None, nondeterministic=False, max_episode_steps=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self.reward_threshold = reward_threshold
        self.nondeterministic = nondeterministic
        self.max_episode_steps = max_episode_steps
        self._kwargs = {} if kwargs is None else kwargs

        self._env_name = id

    def __repr__(self):
        return "EnvSpec({})".format(self.id)
