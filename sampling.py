"""Losowanie bez powtórzeń z trzech części rankingu przebiegów."""
import random


def ranking_key(result, lane):
    return (bool(result['won']), result['reward'], result['score'], -result['steps'], -lane)


def validate_sampling(batch_size, percentages, counts):
    if len(percentages) != 3 or any(type(p) is not int or not 0 <= p <= 100 for p in percentages) or sum(percentages) != 100:
        raise ValueError('Podział: trzy całkowite procenty 0–100, o sumie 100%.')
    if len(counts) != 3 or any(type(n) is not int or n < 0 for n in counts) or not sum(counts):
        raise ValueError('Do BPTT: trzy nieujemne liczby całkowite, łącznie co najmniej 1 przebieg.')
    sizes = [batch_size * p // 100 for p in percentages]
    order = sorted(range(3), key=lambda i: (-(batch_size * percentages[i] % 100), i))
    for i in order[:batch_size - sum(sizes)]:
        sizes[i] += 1
    for i, (size, count) in enumerate(zip(sizes, counts), 1):
        if count > size:
            raise ValueError(f'Grupa {i} ma {size} robotów; nie można wylosować {count} bez powtórzeń. Zwiększ batch lub zmień ustawienia.')
    return sizes


def sample_ranked(results, percentages, counts, seed):
    sizes = validate_sampling(len(results), percentages, counts)
    ranked = sorted(range(len(results)), key=lambda i: ranking_key(results[i], i), reverse=True)
    pools, selected, offset = [], [], 0
    rng = random.Random(seed)
    for size, count in zip(sizes, counts):
        pool = ranked[offset:offset + size]
        pools.append(pool)
        selected.extend(rng.sample(pool, count))
        offset += size
    return selected, pools
