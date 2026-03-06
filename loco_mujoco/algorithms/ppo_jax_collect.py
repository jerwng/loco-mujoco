import numpy as np
import jax
import jax.numpy as jnp
import warnings
from .ppo_jax import PPOJax
from .ppo_jax import PPOAgentConf, PPOAgentState

class PPOJaxCollect(PPOJax):
    @classmethod
    def play_policy(cls, env,
                    agent_conf: PPOAgentConf,
                    agent_state: PPOAgentState,
                    n_envs: int, n_steps=1000, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None, slice_obs=None):

        if use_mujoco and wrap_env:
            if hasattr(agent_conf.experiment, "len_obs_history"):
                assert agent_conf.experiment.len_obs_history == 1, "len_obs_history must be 1 for mujoco envs."
        if use_mujoco:
            assert n_envs == 1, "Only one mujoco env can be run at a time."

        def sample_actions(ts, obs, _rng):
            y, updates = agent_conf.network.apply({'params': ts.params,
                                                   'run_stats': ts.run_stats},
                                                  obs, mutable=["run_stats"])
            ts = ts.replace(run_stats=updates['run_stats'])  # update stats
            pi, _ = y
            a = pi.sample(seed=_rng)
            return a, ts

        config = agent_conf.config.experiment
        train_state = agent_state.train_state

        if deterministic:
            train_state.params["log_std"] = np.ones_like(train_state.params["log_std"]) * -np.inf

        if config.n_seeds > 1:
            assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
                                                  "train_state_seed for replay.")

            # take the seed queried for evaluation
            train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

        if not render and n_steps is None and not record:
            warnings.warn("No rendering, no record, no n_steps specified. This will run forever with no effect.")

        # create env
        if wrap_env and not use_mujoco:
            env = cls._wrap_env(env, config)

        if rng is None:
            rng = jax.random.key(0)

        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        plcy_call = jax.jit(sample_actions)

        # reset env
        if use_mujoco:
            obs = env.reset()
            env_state = None
        else:
            obs, env_state = env.reset(env_keys)
        
        episode_data = []

        done = False

        while not done:
            if use_mujoco:
                frame = env.render(record=record)
            else:
                frame = env.mjx_render(env_state, record=record)
            
            # SAMPLE ACTION
            rng, _rng = jax.random.split(rng)
            action, train_state = plcy_call(train_state, obs, _rng)
            action = jnp.atleast_2d(action)

            if slice_obs is not None:
                obs = obs[:, slice_obs]
            
            episode_data.append({
                'image': frame, # verified dtype is uint8
                'action': np.array(action, dtype=np.float32),
                'obs': np.array(obs, dtype=np.float32), # Only the first 49 obs are related to robot joint position and velocity
                'language_instruction': "walk forward"
            })

            # STEP ENV
            if use_mujoco:
                obs, reward, absorbing, done, info = env.step(action)
            else:
                obs, reward, absorbing, done, info, env_state = env.step(env_state, action)

            # RESET MUJOCO ENV (MJX resets by itself)
            if done:
                if use_mujoco:
                    obs = env.reset()

        env.stop()

        return episode_data

    @classmethod
    def play_policy_mujoco(cls, env,
                           agent_conf: PPOAgentConf,
                           agent_state: PPOAgentState,
                           n_steps=None, render=True,
                           record=False, rng=None, deterministic=False,
                           train_state_seed=None, slice_obs=None):

        return cls.play_policy(env, agent_conf, agent_state, 1, n_steps, render, record, rng, deterministic,
                        True, False, train_state_seed, slice_obs)