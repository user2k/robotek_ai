"""Dalszy trening DQN na losowych labiryntach i ocena na osobnych mapach."""

from collections import deque
from copy import deepcopy
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from queue import Empty
import random
from math import exp
from time import perf_counter
import torch
from batching import run_group, rollout_seed
from sampling import validate_sampling, sample_ranked

from visual_agent import DEFAULT_MODEL, VisualAgent, MOVEMENT
from camera import CAMERA_ANGLE, CAMERA_DEPTH
from episode import Episode
from maps import generate_maze, DEFAULT_FIELDS, validate_fields, MAX_MAP_SIZE
from curriculum import prepare_curriculum, level_rows, advance_level, load_plan, record_streak, level_options

VALIDATION_SEEDS = tuple(-2000001 - 2 * i for i in range(100))
VALIDATION_INTERVAL = 1000
TEST_SEEDS = tuple(-1001 - 2 * i for i in range(16))


def latest_path(path):
    path = Path(path)
    return path.with_name(path.stem + ".latest" + path.suffix)


def training_seed(seed, episode_number):
    # Parzyste, nieujemne seedy treningu nie pokrywają się z zestawami oceny.
    return random.Random(f"{seed}:{episode_number}").getrandbits(62) * 2


def training_size(width, height, recent_average):
    """Jeden awans po pełnym oknie wygranych; maksymalnie 255×255."""
    increase = 2 if recent_average == 1 else 0
    return min(MAX_MAP_SIZE, width + increase), min(MAX_MAP_SIZE, height + increase)


def advance_curriculum(curriculum, won):
    """Zwróć skuteczność bieżącego poziomu i przygotuj następną próbę."""
    if 'plan' in curriculum:
        return advance_level(curriculum, won)
    wins = curriculum['wins']
    average = record_streak(wins, won, 20)
    if len(wins) == 20 and average == 1:
        size = training_size(curriculum['width'], curriculum['height'], average)
        if size != (curriculum['width'], curriculum['height']):
            curriculum['width'], curriculum['height'] = size
            wins.clear()
    return average


def evaluate(agent, rows=None, stop=None, episode_options=None):
    episode = Episode(rows, **(episode_options or {}))
    agent.reset_memory()
    while not episode.done:
        if stop is not None and stop.is_set():
            raise InterruptedError("Walidacja zatrzymana")
        episode.step(agent.act(agent.observe(episode)))
    return episode.metrics()


def evaluate_suite(agent, seeds=VALIDATION_SEEDS, width=11, height=11, stop=None, episode_options=None, pola=DEFAULT_FIELDS):
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("Zestaw oceny nie może być pusty")
    results = [{"map_seed": seed, **evaluate(agent, generate_maze(width, height, seed,
               max_time=(episode_options or {}).get('max_time', 200.), pola=pola), stop=stop,
               episode_options=dict(episode_options or {}, start_seed=seed))}
               for seed in seeds]
    return {"maps": len(results), "wins": sum(r["won"] for r in results),
            "success_rate": sum(r["won"] for r in results) / len(results),
            "score": sum(r["score"] for r in results) / len(results),
            "damage": sum(r["damage"] for r in results) / len(results),
            "time": sum(r["time"] for r in results) / len(results), "results": results}


def validation_options(agent):
    state = agent.curriculum or {}
    return (level_options(state['plan'][state['level']]) if 'plan' in state
            else dict(max_time=200., move_distance=MOVEMENT[1], turn_degrees=MOVEMENT[2], start_direction='right'))


def run_validation(agent, model_path=DEFAULT_MODEL, stop=None, episode_options=None,
                   checkpoint_path='', session='', on_status=None, validation_kind='manual', pola=None):
    """Stały osobny zestaw map, jeden robot, bez uczenia i eksploracji."""
    options = episode_options or agent.training_config.get('episode_options') or validation_options(agent)
    if pola is None:
        state = agent.curriculum or {}
        pola = (agent.training_config.get('pola') or
                (state['plan'][state['level']].get('pola', DEFAULT_FIELDS) if 'plan' in state else DEFAULT_FIELDS))
    pola = validate_fields(pola)
    mode, hidden, previous = agent.network.training, agent.hidden, agent.previous_action
    rng = agent.random.getstate()
    agent.network.eval()
    try:
        with torch.inference_mode():
            result = evaluate_suite(agent, stop=stop, episode_options=options, pola=pola)
    finally:
        agent.network.train(mode)
        agent.hidden, agent.previous_action = hidden, previous
        agent.random.setstate(rng)
    result.update(episodes=agent.episodes, kind=validation_kind,
                  timestamp=datetime.now().isoformat(timespec='seconds'), pola=pola, episode_options=dict(options))
    path = Path(model_path).with_name(Path(model_path).stem + '.validation.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['timestamp', 'session', 'episodes', 'updates', 'rollouts', 'checkpoint',
              'map_index', 'map_seed', 'width', 'height', 'move_distance', 'turn_degrees',
              'max_time', 'start_direction', 'epsilon', 'won', 'damage', 'time', 'score',
              'steps', 'outcome', 'reward', 'maps', 'wins', 'success_rate']
    with path.open('a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if file.tell() == 0:
            writer.writeheader()
        for index, metrics in enumerate(result['results'], 1):
            writer.writerow(dict(timestamp=datetime.now().isoformat(timespec='seconds'), session=session,
                episodes=agent.episodes, updates=agent.updates, rollouts=agent.rollouts,
                checkpoint=str(checkpoint_path), map_index=index, width=11, height=11, epsilon=0.,
                **{k: v for k, v in options.items() if k in fields}, **metrics, maps=result['maps'], wins=result['wins'], success_rate=result['success_rate']))
    # Jeden trwały wpis na ukończoną walidację; nie zapisujemy przerwanych ocen.
    summary_path = Path(model_path).with_name(Path(model_path).stem + '.validation.jsonl')
    with summary_path.open('a', encoding='utf-8') as file:
        file.write(json.dumps({k: v for k, v in result.items() if k != 'results'}, ensure_ascii=False) + '\n')
    if on_status:
        on_status(f"Walidacja po {agent.episodes} grupach: {result['wins']}/100 wygranych · wynik {result['score']:.1f}")
    return result


def train(episodes=300, seed=7, model_path=DEFAULT_MODEL, progress=None, stop=None,
          resume=True, width=15, height=9, on_step=None, device="auto", batch_size=5,
          on_status=None, plan=None, start_level=None, selection=None,
          validation_requests=None, on_validation=None, interventions=None):
    if episodes <= 0:
        raise ValueError("Liczba grup musi być dodatnia")
    if type(batch_size) is not int or not 1 <= batch_size <= 1024:
        raise ValueError("Batch musi wynosić od 1 do 1024")
    if start_level is not None and plan is None:
        raise ValueError('Wybór poziomu startowego wymaga planu poziomów.')
    if selection is not None:
        selection = {'percentages': list(selection['percentages']), 'counts': list(selection['counts'])}
        validate_sampling(batch_size, selection['percentages'], selection['counts'])
    generate_maze(width, height, seed, pola=DEFAULT_FIELDS)
    model_path = Path(model_path)
    latest = latest_path(model_path)
    source = latest if resume and latest.exists() else model_path
    resumed = resume and source.exists()
    agent = VisualAgent.load(source, training=True, device=device) if resumed else VisualAgent(seed=seed, device=device)
    agent.training_selection = selection
    agent.training_config = dict(seed=seed, batch_size=batch_size, selection=deepcopy(selection),
                                 device=str(agent.device), validation_interval=VALIDATION_INTERVAL,
                                 validation_seeds=list(VALIDATION_SEEDS), validation_size=[11, 11])
    start_episode = agent.episodes
    if plan is not None:
        agent.curriculum = prepare_curriculum(agent.curriculum, plan, start_level=start_level)
    if agent.curriculum is None:
        agent.curriculum = {"width": 5, "height": 5, "wins": []}
    if agent.curriculum.get('unit') != 'groups':
        agent.curriculum['wins'] = []
        agent.curriculum['unit'] = 'groups'
    agent.training_config['episode_options'] = validation_options(agent)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not resumed or not latest.exists():
        agent.save(latest)
    history = deque(agent.recent_group_wins, maxlen=20)
    completed_episodes = 0
    error = None
    session = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    def validate_pending(options):
        if validation_requests is None:
            return
        for _ in range(validation_requests.qsize()):
            if stop is not None and stop.is_set():
                raise InterruptedError('Walidacja zatrzymana')
            try:
                validation_requests.get_nowait()
            except Empty:
                return
            result = run_validation(agent, model_path, stop=stop, episode_options=options,
                                    session=session, on_status=on_status,
                                    pola=(agent.curriculum['plan'][agent.curriculum['level']].get('pola', DEFAULT_FIELDS)
                                          if 'plan' in agent.curriculum else DEFAULT_FIELDS))
            if on_validation:
                on_validation(result)
    run_dir = model_path.parent / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / f"{model_path.stem}-{session}.csv"
    rollout_path = csv_path.with_name(csv_path.stem + '.rollouts.csv')
    fields = ["episode", "map_seed", "width", "height", "epsilon", "won", "damage",
              "time", "score", "steps", "outcome", "reward", "batch_size", "winner",
              "rollout_seed", "batch_success_rate", "loss", "steps_per_second", "learn_seconds", "bptt_count"]
    with csv_path.open("w", newline="", encoding="utf-8") as file, \
            rollout_path.open('w', newline='', encoding='utf-8') as rollouts_file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        all_writer = csv.DictWriter(rollouts_file, fieldnames=["episode", "map_seed", "lane", "rollout_seed",
                                   "selected", "rank_group", "won", "damage", "time", "score", "steps", "outcome", "reward"])
        all_writer.writeheader()
        for number in range(1, episodes + 1):
            if agent.curriculum.get('completed') or (stop is not None and stop.is_set()):
                break
            try:
                validate_pending(validation_options(agent))
            except InterruptedError:
                break
            episode_number = start_episode + number
            map_seed = training_seed(seed, episode_number)
            map_width, map_height = agent.curriculum['width'], agent.curriculum['height']
            stage = agent.curriculum.get('level', 0) + 1
            pola = (agent.curriculum['plan'][stage - 1].get('pola', DEFAULT_FIELDS)
                    if 'plan' in agent.curriculum else DEFAULT_FIELDS)
            agent.training_config['pola'] = pola
            rows = (level_rows(agent.curriculum, map_seed) if 'plan' in agent.curriculum
                    else generate_maze(map_width, map_height, map_seed, pola=pola))
            episode_options = (level_options(agent.curriculum['plan'][stage - 1])
                               if 'plan' in agent.curriculum else {})
            agent.training_config['episode_options'] = deepcopy(episode_options or validation_options(agent))
            seeds = [rollout_seed(seed, episode_number, i) for i in range(batch_size)]
            epsilon = 0.1 + 0.6 * exp(-(episode_number - 1) / 250)
            if on_status:
                on_status(f"Symulacja · {agent.device} · grupa {episode_number} · batch {batch_size}")
            try:
                agent.reset_memory()
                group = run_group(agent, [Episode(rows, **dict(episode_options, start_seed=lane_seed))
                                         if episode_options else Episode(rows) for lane_seed in seeds], seeds, epsilon,
                                  stop=stop, on_step=on_step, on_status=on_status, interventions=interventions,
                                  metadata={"episode": episode_number, "session_episode": number,
                                            "map_seed": map_seed, "epsilon": epsilon, "device": str(agent.device), "level": stage,
                                            "level_name": agent.curriculum["plan"][stage - 1].get("name", f"Poziom {stage}") if "plan" in agent.curriculum else ""})
                if group is None or (stop is not None and stop.is_set()):
                    break
                winner = group['winner']
                selected, pools = [winner], []
                if selection is not None:
                    selected, pools = sample_ranked(group['results'], selection['percentages'],
                                                    selection['counts'], f'{seed}:{episode_number}:bptt')
                if on_status:
                    on_status(f"Full-episode BPTT · {agent.device} · {len(selected)} przebiegów · "
                              "osobne historie GRU · średnia strat · jeden krok optymalizatora")
                started = perf_counter()
                loss = (agent.learn_episodes([group['trajectories'][i] for i in selected])
                        if selection is not None else agent.learn_episode(group['trajectory']))
                learn_seconds = perf_counter() - started
            except (torch.cuda.OutOfMemoryError, MemoryError):
                agent.optimizer.zero_grad(set_to_none=True)
                if agent.device.type == 'cuda':
                    torch.cuda.empty_cache()
                error = "Za mało pamięci. Zachowano ostatni zapis dyskowy; niezapisane grupy trzeba powtórzyć. Zmniejsz batch."
                break
            agent.episodes = episode_number
            agent.rollouts += batch_size
            # Jedna grupa wnosi jeden wynik wybranego najlepszego robota.
            stage_success = advance_curriculum(agent.curriculum, group['results'][winner]['won'])
            for lane, result in enumerate(group['results']):
                all_writer.writerow({"episode": episode_number, "map_seed": map_seed, "lane": lane + 1,
                                     "rollout_seed": seeds[lane], "selected": lane in selected,
                                     "rank_group": next((i + 1 for i, pool in enumerate(pools) if lane in pool), ""), **result})
            rollouts_file.flush()
            row = {"episode": episode_number, "map_seed": map_seed, "width": map_width, "height": map_height,
                   "epsilon": epsilon, **group['results'][winner], "batch_size": batch_size, "winner": winner + 1,
                   "rollout_seed": seeds[winner],
                   "batch_success_rate": sum(r['won'] for r in group['results']) / batch_size,
                   "loss": loss, "steps_per_second": group['steps_per_second'], "learn_seconds": learn_seconds, "bptt_count": len(selected)}
            writer.writerow(row)
            file.flush()
            history.append(bool(row["won"]))
            agent.recent_group_wins = list(history)
            completed_episodes += 1
            report = {**row, "session_episode": number,
                      "success_rate": sum(history) / len(history),
                      "recent_group_wins": sum(history), "recent_group_count": len(history),
                      "stage_success_rate": stage_success, "next_width": agent.curriculum["width"],
                      "next_height": agent.curriculum["height"]}
            if 'plan' in agent.curriculum:
                state = agent.curriculum
                report.update(level=stage, next_level=state['level'] + 1,
                              level_count=len(state['plan']), completed=state['completed'],
                              level_wins=sum(state['wins']),
                              required=state['plan'][state['level']]['required'])
            if number % 10 == 0 or number == episodes or agent.curriculum.get('completed'):
                agent.save(latest)
            if progress:
                progress(report)
            del group
            if episode_number % VALIDATION_INTERVAL == 0:
                snapshot = model_path.parent / 'checkpoints' / f'{model_path.stem}-{session}-episode-{episode_number}.pt'
                agent.training_config['episode_options'] = deepcopy(episode_options or validation_options(agent))
                agent.save(snapshot)
                if on_status:
                    on_status(f"Walidacja · 100 map 11×11 · po {episode_number} grupach · jeden robot · epsilon 0")
                try:
                    result = run_validation(agent, model_path, stop, episode_options or validation_options(agent),
                                   checkpoint_path=snapshot, session=session, on_status=on_status, validation_kind='periodic', pola=pola)
                    if on_validation:
                        on_validation(result)
                except InterruptedError:
                    break
        if error is None and not (stop is not None and stop.is_set()):
            try:
                validate_pending(validation_options(agent))
            except InterruptedError:
                pass
    if error is None:
        if on_status:
            on_status("Zapisuję najnowszy model…")
        agent.save(latest)
    report = {"training_episodes": completed_episodes, "total_episodes": agent.episodes,
              "training_rollouts": completed_episodes * batch_size, "total_rollouts": agent.rollouts,
              "batch_size": batch_size, "training_mode": "stratified" if selection else "best_of_batch", "selection": selection, "error": error,
              "started_from_episode": start_episode, "resumed": resumed,
              "selected_episode": agent.episodes, "seed": seed,
              "architecture": "cnn_gru", "hidden_size": agent.hidden_size,
              "device": str(agent.device), "camera": [80, 60, CAMERA_ANGLE, CAMERA_DEPTH],
              "movement": MOVEMENT, "actions": 4, "bptt": "full_episode", "curriculum": agent.curriculum,
              "width": width, "height": height, "history": str(csv_path), "rollout_history": str(rollout_path),
              "evaluation_scope": "Co 1000 grup: 100 stałych map testowych 11×11, jeden robot, epsilon 0.",
              "validation_history": str(model_path.with_name(model_path.stem + '.validation.csv'))}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    model_path.with_suffix(".json").write_text(text, encoding="utf-8")
    csv_path.with_suffix(".json").write_text(text, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300, help="Liczba dodatkowych grup/map; każda zawiera batch robotów")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--batch-size", type=int, default=5, help="Liczba robotów na wspólnej mapie (1–1024)")
    parser.add_argument("--split", type=int, nargs=3, help="Procenty trzech grup rankingu, np. 10 20 70")
    parser.add_argument("--samples", type=int, nargs=3, help="Liczba przebiegów do BPTT z każdej grupy, np. 3 2 3")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--start-level", type=int, help="Zacznij wskazany poziom od zera (wymaga --plan)")
    parser.add_argument("--plan", type=Path, help="Plik JSON z planem poziomów")
    parser.add_argument("--fresh", action="store_true", help="Zacznij od nowych wag zamiast kontynuować")
    parser.add_argument("--evaluate", action="store_true", help="Oceń najnowszy model na mapach testowych")
    args = parser.parse_args()
    if (args.split is None) != (args.samples is None):
        parser.error("Podaj razem --split i --samples")
    if args.evaluate:
        source = latest_path(args.model) if latest_path(args.model).exists() else args.model
        result = run_validation(VisualAgent.load(source, device=args.device), args.model, checkpoint_path=source)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        def progress(row):
            if row["session_episode"] % 10 == 0:
                print(f"Epizod {row['episode']} (+{row['session_episode']}): "
                      f"sukcesy(20)={row['success_rate']:.0%}", flush=True)
        report = train(args.episodes, args.seed, args.model, progress,
                       resume=not args.fresh, width=args.width, height=args.height, device=args.device, batch_size=args.batch_size,
                       plan=load_plan(args.plan) if args.plan else None, start_level=args.start_level,
                       selection={'percentages': args.split, 'counts': args.samples} if args.split is not None else None)
        print(json.dumps({key: report[key] for key in ("training_episodes", "total_episodes", "resumed",
                         "selected_episode", "history")}, ensure_ascii=False, indent=2))



if __name__ == "__main__":
    main()
