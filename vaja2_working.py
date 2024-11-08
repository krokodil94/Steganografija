
def select_random_triplets(zigzag_blocks, N, M):
    random.seed(N * M)  # Setting the random seed for repeatability
    selected_triplets = []


    for block in zigzag_blocks:
        block_triplets = []
        used_indices = set()

        while len(block_triplets) < M:
            start_index = random.randint(4, upper_limit - 3)

            # Ensure no overlapping
            if not any(index in used_indices for index in range(start_index, start_index + 3)):
                block_triplets.append((start_index, start_index + 1, start_index + 2))
                used_indices.update(range(start_index, start_index + 3))

        selected_triplets.append(block_triplets)

    return selected_triplets

M = 5
selected_triplets = select_random_triplets(zigzag_blocks, N, M)
print(selected_triplets)