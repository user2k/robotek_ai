"""Wspólne wagi, jedna mapa i niezależne stany GRU/RNG w batchu."""
import random
from queue import Empty
from time import perf_counter
import torch
from visual_agent import START_ACTION
from tensor_camera import TensorCamera, camera_pose


def rollout_seed(seed, group_number, lane):
    return random.Random(f'rollout:{seed}:{group_number}:{lane}').getrandbits(63)


def best_index(episodes):
    # END ma priorytet, potem pełna nagroda; wynik/krótsza droga rozstrzygają remis.
    return max(range(len(episodes)), key=lambda i: (
        episodes[i].player.won, episodes[i].total_reward,
        episodes[i].score, -episodes[i].steps, -i))


def run_group(agent, episodes, seeds, epsilon, stop=None, on_step=None, metadata=None, on_status=None, interventions=None):
    if not episodes or len(episodes) != len(seeds):
        raise ValueError('Batch wymaga epizodów i jednego seeda na każdego robota')
    count = len(episodes)
    generators = [random.Random(seed) for seed in seeds]
    camera = TensorCamera(episodes[0].world, agent.device)
    trajectories = [dict(poses=[camera_pose(ep.player)], camera=camera, actions=[], rewards=[], dones=[]) for ep in episodes]
    previous = [START_ACTION] * count
    hidden = torch.zeros(1, count, agent.hidden_size, device=agent.device)
    active = list(range(count))
    steps = 0
    start = perf_counter()
    base = metadata or {}
    last_status = float("-inf")
    last_actions = [None] * count
    last_rewards = [0.] * count

    def publish(lane, action, reward):
        nonlocal last_status
        now = perf_counter()
        if on_status is not None and (now - last_status >= .5 or steps == count or not active):
            player = episodes[lane].player
            label = "START" if action is None else [f"przód {player.move_distance:g} m", f"lewo {player.turn_degrees:g}°", f"prawo {player.turn_degrees:g}°", f"tył {player.move_distance:g} m"][action]
            on_status(f"Symulacja · {base.get('device', agent.device)} · grupa {base.get('episode', 1)} · "
                      f"robot {lane + 1}/{count}: krok {episodes[lane].steps} ({label}) · "
                      f"łącznie {steps} ruchów · ukończono {count - len(active)}/{count} · "
                      f"{steps / max(now - start, 1e-9):.0f} kroków/s")
            last_status = now
        if on_step is not None:
            on_step(episodes[lane], {**base, 'action': action, 'reward': reward,
                    'lane': lane + 1, 'batch_size': count, 'active': len(active),
                    'rollout_seed': seeds[lane], 'total_steps': steps,
                    'bots': episodes,
                    'bot_actions': last_actions, 'bot_rewards': last_rewards,
                    'steps_per_second': steps / max(perf_counter() - start, 1e-9)})

    publish(0, None, 0.)
    while active:
        if stop is not None and stop.is_set():
            return None
        if interventions is not None:
            while True:
                try:
                    group_number, lane = interventions.get_nowait()
                except Empty:
                    break
                # Stare kliknięcie nie może zabić robota z nowej grupy.
                if group_number != base.get('episode') or lane not in active:
                    continue
                trace = trajectories[lane]
                # START nie ma jeszcze przejścia do uczenia.
                if not trace['actions']:
                    interventions.put((group_number, lane))
                    break
                if episodes[lane].swat():
                    trace['rewards'][-1] -= 20.
                    trace['dones'][-1] = True
                    last_rewards[lane] = -20.
            active = [i for i in active if not episodes[i].done]
            if not active:
                publish(0, None, -20.)
                break
        states = camera.render([camera_pose(episodes[i].player) for i in active])
        actions, next_hidden = agent.act_batch(
            states, [previous[i] for i in active],
            hidden[:, active, :], [generators[i] for i in active], epsilon)
        hidden[:, active, :] = next_hidden
        watching = active[0]
        shown_action, shown_reward = None, 0.
        for lane, action in zip(active, actions):
            if stop is not None and stop.is_set():
                return None
            result = episodes[lane].step(action)
            last_actions[lane], last_rewards[lane] = action, result.reward
            trace = trajectories[lane]
            trace['poses'].append(camera_pose(episodes[lane].player))
            trace['actions'].append(action)
            trace['rewards'].append(result.reward)
            trace['dones'].append(result.done)
            previous[lane] = action
            agent.env_steps += 1
            steps += 1
            if lane == watching:
                shown_action, shown_reward = action, result.reward
        active = [i for i in active if not episodes[i].done]
        publish(watching, shown_action, shown_reward)
    if stop is not None and stop.is_set():
        return None
    winner = best_index(episodes)
    return {'winner': winner, 'trajectory': trajectories[winner], 'trajectories': trajectories,
            'results': [episode.metrics() for episode in episodes],
            'steps_per_second': steps / max(perf_counter() - start, 1e-9)}
