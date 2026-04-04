import numpy as np
import jax
import jax.numpy as jnp
import warnings
import mujoco as _mujoco
from .ppo_jax_collect import PPOJaxCollect

# Geom names that represent normal foot-floor walking contacts (not obstacle collisions)
_FOOT_GEOM_NAMES = frozenset({'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'})
_FLOOR_GEOM_NAMES = frozenset({'floor'})


def _get_allowed_geom_ids(model):
    """Return set of geom IDs for foot and floor geoms (normal walking contacts)."""
    ids = set()
    for name in _FOOT_GEOM_NAMES | _FLOOR_GEOM_NAMES:
        gid = _mujoco.mj_name2id(model, _mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            ids.add(gid)
    return ids


def _check_obstacle_collision_mujoco(model, data, allowed_geom_ids):
    """Return (collided, g1_name, g2_name) if a non-foot-floor contact exists."""
    for i in range(data.ncon):
        g1 = int(data.contact[i].geom1)
        g2 = int(data.contact[i].geom2)
        if not (g1 in allowed_geom_ids and g2 in allowed_geom_ids):
            return True, model.geom(g1).name, model.geom(g2).name
    return False, None, None


def _check_obstacle_collision_mjx(model, env_state, allowed_geom_ids):
    """Return (collided, g1_name, g2_name) for non-foot-floor contact in MjX."""
    try:
        contact_geom = env_state.data.contact.geom
        contact_dist = env_state.data.contact.dist
        # Handle batched shape (n_envs, max_contacts, 2)
        if len(contact_geom.shape) == 3:
            geom_arr = np.array(contact_geom[0])
            dist_arr = np.array(contact_dist[0])
        else:
            geom_arr = np.array(contact_geom)
            dist_arr = np.array(contact_dist)
        for i in range(len(dist_arr)):
            if float(dist_arr[i]) >= 0:  # inactive contact slot
                continue
            g1 = int(geom_arr[i, 0])
            g2 = int(geom_arr[i, 1])
            if not (g1 in allowed_geom_ids and g2 in allowed_geom_ids):
                return True, model.geom(g1).name, model.geom(g2).name
    except Exception:
        pass
    return False, None, None


def _get_target_body_id(model, target_body_name):
    """Return body ID for target_body_name, or -1 if not found."""
    return _mujoco.mj_name2id(model, _mujoco.mjtObj.mjOBJ_BODY, target_body_name)


def _get_target_xy(data, body_id, env_state, use_mujoco):
    """Return [x, y] world position of target body, or None if body_id < 0."""
    if body_id < 0:
        return None
    try:
        if use_mujoco:
            return np.array(data.xpos[body_id, :2])
        else:
            xpos = env_state.data.xpos
            if len(xpos.shape) == 3:  # (n_envs, nbody, 3)
                return np.array(xpos[0, body_id, :2])
            else:
                return np.array(xpos[body_id, :2])
    except Exception:
        return None


def _print_episode_result(
    step_count, planar_speed, current_qpos, env, env_state,
    target_body_id, target_body_name,
    collision_occurred, collision_geoms,
    success_velocity_threshold, success_distance_threshold,
    use_mujoco,
):
    """Print episode success/failure evaluation."""
    robot_xy = np.array([float(current_qpos[0]), float(current_qpos[1])])
    if use_mujoco:
        target_xy = _get_target_xy(env.data, target_body_id, None, True)
    else:
        target_xy = _get_target_xy(None, target_body_id, env_state, False)

    dist = float(np.linalg.norm(robot_xy - target_xy)) if target_xy is not None else None

    if collision_occurred:
        outcome = "FAILURE"
        reason = f"Collided with obstacle ({collision_geoms[0]} <-> {collision_geoms[1]})"
    elif planar_speed < success_velocity_threshold and (dist is None or dist < success_distance_threshold):
        outcome = "SUCCESS"
        if dist is not None:
            reason = f"Reached target (speed={planar_speed:.3f} m/s, dist={dist:.3f} m)"
        else:
            reason = f"Low velocity stop (speed={planar_speed:.3f} m/s, target '{target_body_name}' not in model)"
    else:
        outcome = "TIMEOUT"
        parts = [f"speed={planar_speed:.3f} m/s (threshold={success_velocity_threshold})"]
        if dist is not None:
            parts.append(f"dist={dist:.3f} m (threshold={success_distance_threshold} m)")
        reason = ", ".join(parts)

    print("\n" + "=" * 60)
    print("EPISODE EVALUATION")
    print("=" * 60)
    print(f"Outcome : {outcome}")
    print(f"Reason  : {reason}")
    print(f"Steps   : {step_count}")
    print(f"Speed   : {planar_speed:.3f} m/s  (threshold: {success_velocity_threshold} m/s)")
    if dist is not None:
        print(f"Distance: {dist:.3f} m  (threshold: {success_distance_threshold} m, target: '{target_body_name}')")
    else:
        print(f"Distance: N/A  (target body '{target_body_name}' not found in model)")
    print(f"Collision detected: {'YES — ' + collision_geoms[0] + ' <-> ' + collision_geoms[1] if collision_occurred else 'No'}")
    print("=" * 60)

class PPOJaxCollectVLMRL(PPOJaxCollect):
    """
    Custom PPOJax collector for Go2 that allows overriding goal parameters.
    The goal observation is assumed to be the last 3 elements: [vel_x, vel_y, heading]
    """
    
    @classmethod
    def play_policy(cls, env,
                    agent_conf,
                    agent_state,
                    n_envs: int, n_steps=1000, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None, slice_obs=None,
                    custom_goal=None, vlm_predictor=None,
                    vlm_update_frequency=1, vlm_prompt=None,
                    low_velocity_threshold=None,
                    low_velocity_window=50,
                    target_body_name="red_disk_marker",
                    success_distance_threshold=1.0,
                    success_velocity_threshold=0.1):
        """
        Play policy with optional custom goal override or VLM-based goal prediction.
        
        Args:
            custom_goal: Optional dict with keys 'vel_x', 'vel_y', 'heading' to override the goal.
                        If None and vlm_predictor is None, uses default goal.
            vlm_predictor: Optional VLMGoalPredictor instance for dynamic goal prediction from frames
            vlm_update_frequency: How often to update VLM prediction (in steps)
            vlm_prompt: Optional text prompt for VLM
        """

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
        def quat_to_heading(quat):
            """Convert quaternion [w, x, y, z] to heading (yaw) angle in radians."""
            w, x, y, z = quat[0], quat[1], quat[2], quat[3]
            # Yaw (heading) from quaternion
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            return yaw
        
        if use_mujoco:
            obs = env.reset()
            env_state = None
            # Print initial position from qpos
            qpos = env.data.qpos
            quat = qpos[3:7]  # [w, x, y, z]
            heading = quat_to_heading(quat)
        else:
            obs, env_state = env.reset(env_keys)
            # Print initial position from qpos
            qpos = env_state.data.qpos[0] if len(env_state.data.qpos.shape) > 1 else env_state.data.qpos
            quat = qpos[3:7]  # [w, x, y, z]
            heading = quat_to_heading(quat)
        
        print(f"Initial position - X: {qpos[0]:.3f}, Y: {qpos[1]:.3f}, Z: {qpos[2]:.3f}")
        print(f"Initial orientation (quaternion) - W: {quat[0]:.3f}, X: {quat[1]:.3f}, Y: {quat[2]:.3f}, Z: {quat[3]:.3f}")
        print(f"Initial heading: {heading:.3f} rad ({np.degrees(heading):.1f}°)")
        print("-" * 50)

        # Determine goal source
        use_vlm = vlm_predictor is not None
        use_custom = custom_goal is not None and not use_vlm
        
        if use_vlm:
            print(f"Using VLM predictor for dynamic goals (update frequency: {vlm_update_frequency} steps)")

            # Get initial frame for VLM prediction
            if use_mujoco:
                initial_frame = env.render(record=record)
            else:
                initial_frame = env.mjx_render(env_state, record=record)
            
            # Get initial VLM prediction (robot is stationary at episode start)
            vlm_goal = vlm_predictor.predict_goal(
                initial_frame, prompt=vlm_prompt, current_heading=heading,
            )
            goal_values = np.array([
                vlm_goal.get('vel_x', 1.0),
                vlm_goal.get('vel_y', 0.0),
                vlm_goal.get('heading', 0.0)
            ], dtype=np.float32)

            print(f"Initial VLM goal: vel_x={goal_values[0]:.3f}, vel_y={goal_values[1]:.3f}, heading={goal_values[2]:.3f} ({np.degrees(goal_values[2]):.1f}°)")
        
        elif use_custom:
            # Use custom goal if custom_goal is provided
            goal_values = np.array([
                custom_goal.get('vel_x', 0.0),
                custom_goal.get('vel_y', 0.0),
                custom_goal.get('heading', 0.0)
            ], dtype=np.float32)
            print(f"Using custom goal: vel_x={goal_values[0]:.3f}, vel_y={goal_values[1]:.3f}, heading={goal_values[2]:.3f}")
        else:
            # Use default goal
            goal_values = np.array([1, 0, 0], dtype=np.float32)
            print(f"Using default goal: vel_x={goal_values[0]:.3f}, vel_y={goal_values[1]:.3f}, heading={goal_values[2]:.3f}")

        if use_mujoco:
            obs[-3:] = goal_values
        else:
            # For batched observations (n_envs, obs_dim)
            obs = obs.at[:, -3:].set(goal_values)

        # Episode evaluation setup
        raw_model = env.model
        allowed_geom_ids = _get_allowed_geom_ids(raw_model)
        target_body_id = _get_target_body_id(raw_model, target_body_name)
        if target_body_id < 0:
            print(f"[EvalChecker] Target body '{target_body_name}' not found — distance check disabled.")
        else:
            print(f"[EvalChecker] Tracking target body '{target_body_name}' (ID={target_body_id}).")
        collision_occurred = False
        collision_geoms = (None, None)
        # Sentinel values used if the loop never executes
        planar_speed = 0.0
        current_qpos = qpos

        done = False
        step_count = 0
        recent_speeds = []

        print("-" * 50)

        while not done:
            # Check if we've reached the step limit
            if n_steps is not None and step_count >= n_steps:
                print(f"Reached step limit ({n_steps} steps). Terminating.")
                break
            if use_mujoco:
                frame = env.render(record=record)
            else:
                frame = env.mjx_render(env_state, record=record)

            # Update VLM goal if using VLM and it's time to update
            if use_vlm and step_count % vlm_update_frequency == 0 and step_count > 0:
                # Get current heading from environment state
                if use_mujoco:
                    current_qpos = env.data.qpos
                else:
                    current_qpos = env_state.data.qpos[0] if len(env_state.data.qpos.shape) > 1 else env_state.data.qpos
                current_quat = current_qpos[3:7]
                current_heading = quat_to_heading(current_quat)
                vlm_goal = vlm_predictor.predict_goal(
                    frame, prompt=vlm_prompt, current_heading=current_heading,
                )
                goal_values = np.array([
                    vlm_goal.get('vel_x', 1.0),
                    vlm_goal.get('vel_y', 0.0),
                    vlm_goal.get('heading', 0.0)
                ], dtype=np.float32)
                print(f"[Step {step_count}] VLM updated goal: vel_x={goal_values[0]:.3f}, vel_y={goal_values[1]:.3f}, heading={goal_values[2]:.3f} ({np.degrees(goal_values[2]):.1f}°)")
                
                # Update observation with new goal
                if use_mujoco:
                    obs[-3:] = goal_values
                else:
                    obs = obs.at[:, -3:].set(goal_values)
            
            # SAMPLE ACTION
            rng, _rng = jax.random.split(rng)
            action, train_state = plcy_call(train_state, obs, _rng)
            action = jnp.atleast_2d(action)

            # STEP ENV
            if use_mujoco:
                obs, reward, absorbing, done, info = env.step(action)
            else:
                obs, reward, absorbing, done, info, env_state = env.step(env_state, action)
            
            # Get current heading and velocity
            if use_mujoco:
                current_qpos = env.data.qpos
                current_qvel = env.data.qvel
            else:
                current_qpos = env_state.data.qpos[0] if len(env_state.data.qpos.shape) > 1 else env_state.data.qpos
                current_qvel = env_state.data.qvel[0] if len(env_state.data.qvel.shape) > 1 else env_state.data.qvel

            current_quat = current_qpos[3:7]
            current_heading = quat_to_heading(current_quat)
            vel_x = float(current_qvel[0])
            vel_y = float(current_qvel[1])
            planar_speed = float(np.linalg.norm(current_qvel[:2]))
            
            print(f"[Step {step_count}] vel_x: {vel_x:.3f}, vel_y: {vel_y:.3f}, speed: {planar_speed:.3f}, heading: {current_heading:.3f} rad ({np.degrees(current_heading):.1f}°)")
            if use_mujoco:
                obs[-3:] = goal_values
            else:
                obs = obs.at[:, -3:].set(goal_values)

            step_count += 1

            # Check for obstacle collision (only update flag, never clear it)
            if not collision_occurred:
                if use_mujoco:
                    coll, g1, g2 = _check_obstacle_collision_mujoco(raw_model, env.data, allowed_geom_ids)
                else:
                    coll, g1, g2 = _check_obstacle_collision_mjx(raw_model, env_state, allowed_geom_ids)
                if coll:
                    collision_occurred = True
                    collision_geoms = (g1, g2)
                    print(f"[EvalChecker] Collision at step {step_count}: {g1} <-> {g2}")

            if low_velocity_threshold is not None and low_velocity_window > 0:
                recent_speeds.append(planar_speed)
                if len(recent_speeds) > low_velocity_window:
                    recent_speeds.pop(0)

                if len(recent_speeds) == low_velocity_window:
                    avg_speed = float(np.mean(recent_speeds))
                    if avg_speed < low_velocity_threshold:
                        print(
                            f"Average planar speed over the last {low_velocity_window} steps "
                            f"fell below {low_velocity_threshold:.3f} ({avg_speed:.3f}). Terminating."
                        )
                        done = True
            
            # RESET MUJOCO ENV (MJX resets by itself)
            if done:
                if use_mujoco:
                    obs = env.reset()
            
            print()

        env.stop()

        print(f"Episode completed: {step_count} steps")

        _print_episode_result(
            step_count=step_count,
            planar_speed=planar_speed,
            current_qpos=current_qpos,
            env=env,
            env_state=env_state,
            target_body_id=target_body_id,
            target_body_name=target_body_name,
            collision_occurred=collision_occurred,
            collision_geoms=collision_geoms,
            success_velocity_threshold=success_velocity_threshold,
            success_distance_threshold=success_distance_threshold,
            use_mujoco=use_mujoco,
        )

    @classmethod
    def play_policy_mujoco(cls, env,
                           agent_conf,
                           agent_state,
                           n_steps=None, render=True,
                           record=False, rng=None, deterministic=False,
                           train_state_seed=None, slice_obs=None,
                           custom_goal=None, vlm_predictor=None,
                           vlm_update_frequency=1, vlm_prompt=None,
                           low_velocity_threshold=None,
                           low_velocity_window=50,
                           target_body_name="red_disk_marker",
                           success_distance_threshold=1.0,
                           success_velocity_threshold=0.1):

        return cls.play_policy(env, agent_conf, agent_state, 1, n_steps, render, record, rng, deterministic,
                        True, False, train_state_seed, slice_obs, custom_goal, vlm_predictor,
                        vlm_update_frequency, vlm_prompt,
                        low_velocity_threshold, low_velocity_window,
                        target_body_name, success_distance_threshold, success_velocity_threshold)
