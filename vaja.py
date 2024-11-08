

def apply_F5_algorithm_rgb(zigzag_blocks, selected_triplets, binary_message):
    for block_index, block_triplets in enumerate(selected_triplets):
        for triplet_indices in block_triplets:
            # Randomly choose a channel (R, G, or B) for embedding
            channel = random.choice([0, 1, 2])  # 0 for R, 1 for G, 2 for B

            # Extract the RGB tuples using indices from the triplet
            AC1, AC2, AC3 = [zigzag_blocks[block_index][index][channel] for index in triplet_indices]
            C1, C2, C3 = AC1 & 1, AC2 & 1, AC3 & 1  # LSB

            x1, x2 = int(binary_message[0]), int(binary_message[1])
            binary_message = binary_message[2:]  # Remove used bits from the message

            # F5 algorithm for hiding bits
            if x1 != C1 and x2 == (C2 ^ C3):
                AC1 ^= 1
            elif x1 == C1 and x2 != (C2 ^ C3):
                AC3 ^= 1
            elif x1 != C1 and x2 != (C2 ^ C3):
                AC2 ^= 1

            # Replace the processed channel in the RGB tuples
            for i, index in enumerate(triplet_indices):
                r, g, b = zigzag_blocks[block_index][index]
                modified_color = (AC1, AC2, AC3)[i]
                zigzag_blocks[block_index][index] = (r, g, b)[:channel] + (modified_color,) + (r, g, b)[channel+1:]

    return zigzag_blocks

binary_message = "Hello world!"
M = 5
binary_message = binarize_message(binary_message)




def select_random_triplets(N, M):
    # Set the random seed for repeatability
    random.seed(N * M)
    # Adjust the range of indices based on N
    upper_limit = 64 - N
    if upper_limit > 32:
        upper_limit = 32

    # Ensure we have enough indices to choose from
    if (upper_limit - 4) < 3 * M:
        raise ValueError("Not enough indices to select M triplets")

    # List to store the selected triplets
    selected_triplets = []

    # Set of indices to ensure there's no overlapping
    used_indices = set()

    while len(selected_triplets) < M:
        # Generate a random start index
        start_index = random.randint(4, upper_limit - 3)

        # Check if the triplet overlaps with previously selected triplets
        if not any(index in used_indices for index in range(start_index, start_index + 3)):
            # Add the triplet to the list
            selected_triplets.append((start_index, start_index + 1, start_index + 2))

            # Mark these indices as used
            used_indices.update(range(start_index, start_index + 3))

    return selected_triplets
"""binary_message = binarize_message("Hello world!")
N = 40
M = 5
triplets = select_random_triplets(N, M)
print(triplets)"""

def apply_F5_algorithm(triplets, binary_message):
    for triplet in triplets:
        AC1, AC2, AC3 = triplet[0], triplet[1], triplet[2]
        C1, C2, C3 = AC1 & 1, AC2 & 1, AC3 & 1  # LSB
        x1, x2 = int(binary_message[0]), int(binary_message[1])
        binary_message = binary_message[2:]  # Odstranimo uporabljena bita iz sporočila

        # Operacije za skrivanje bitov
        if x1 != C1 and x2 == (C2 ^ C3):
            AC1 ^= 1  # Negacija LSB AC1
        elif x1 == C1 and x2 != (C2 ^ C3):
            AC3 ^= 1  # Negacija LSB AC3
        elif x1 != C1 and x2 != (C2 ^ C3):
            AC2 ^= 1  # Negacija LSB AC2











