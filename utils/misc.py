def build_neighbour_list(N):
    pairs, seen = [], set()
    for r in range(N):
        for c in range(N):
            i = r * N + c
            for nr, nc in [((r + 1) % N, c), (r, (c + 1) % N)]:
                j = nr * N + nc
                if (i, j) not in seen:
                    pairs.append((i, j))
                    seen.add((i, j))
    return pairs


def build_all_pairs(D):
    return [(i, j) for i in range(D) for j in range(i + 1, D)]
