import numpy as np
import math
import imageio
from random import sample, seed, randint
import struct
"""KONSTANTE"""

a = math.sqrt(8 / 64)
b = math.sqrt(2 / 4)
c = 0.5
H = [[a, a, c, 0, b, 0, 0, 0],
     [a, a, c, 0, -b, 0, 0, 0],
     [a, a, -c, 0, 0, b, 0, 0],
     [a, a, -c, 0, 0, -b, 0, 0],
     [a, -a, 0, c, 0, 0, b, 0],
     [a, -a, 0, c, 0, 0, -b, 0],
     [a, -a, 0, -c, 0, 0, 0, b],
     [a, -a, 0, -c, 0, 0, 0, -b]]

array_H = np.array(H)
transpose_H = np.transpose(H)
cikCakOrder = [
        [0, 0],
        [0, 1], [1, 0],
        [2, 0], [1, 1], [0, 2],
        [0, 3], [1, 2], [2, 1], [3, 0],
        [4, 0], [3, 1], [2, 2], [1, 3], [0, 4],
        [0, 5], [1, 4], [2, 3], [3, 2], [4, 1], [5, 0],
        [6, 0], [5, 1], [4, 2], [3, 3], [2, 4], [1, 5], [0, 6],
        [0, 7], [1, 6], [2, 5], [3, 4], [4, 3], [5, 2], [6, 1], [7, 0],
        [7, 1], [6, 2], [5, 3], [4, 4], [3, 5], [2, 6], [1, 7],
        [2, 7], [3, 6], [4, 5], [5, 4], [6, 3], [7, 2],
        [7, 3], [6, 4], [5, 5], [4, 6], [3, 7],
        [4, 7], [5, 6], [6, 5], [7, 4],
        [7, 5], [6, 6], [5, 7],
        [6, 7], [7, 6],
        [7, 7]
    ]

"""ENCODING"""


def read_image(image_path):
    image = imageio.v2.imread(image_path)

    # Check if the image is grayscale or color
    if image.ndim == 2:  # Grayscale image, 2 dimensions
        height, width = image.shape
        height_padding = (8 - height % 8) % 8
        width_padding = (8 - width % 8) % 8
        pad_width = ((0, height_padding), (0, width_padding))
    else:  # Color image, 3 dimensions
        height, width, _ = image.shape
        height_padding = (8 - height % 8) % 8
        width_padding = (8 - width % 8) % 8
        pad_width = ((0, height_padding), (0, width_padding), (0, 0))

    # Apply padding accordingly
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)

    return padded_image, height, width

image, height, width = read_image("Baboon.bmp")
print("Original Image Shape:", image.shape)

def binarize_message(message):
    # Binarizacija sporočila
    binary_message = ''.join(format(ord(char), '08b') for char in message)

    # Dodaj 4 zloge za velikost sporočila v bitih
    binary_message = format(len(binary_message), '032b') + binary_message

    return binary_message
binary_message = binarize_message("Danes je lep dan ")
def process_image_grayscale(image, array_H, transpose_H, zigzag_order, N):
    assert image.ndim == 2, "Image must be a grayscale image"
    zigzag_blocks = []
    height, width = image.shape

    # Process each 8x8 block of the image
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = image[i:i + 8, j:j + 8]

            # Apply the transformation
            transformed_block = np.dot(np.dot(transpose_H, block), array_H)

            # Reorder the transformed block in a zigzag pattern
            zigzag_block = [transformed_block[tuple(element)] for element in zigzag_order]

            # Set the last N coefficients to 0
            for k in range(1, N + 1):
                zigzag_block[-k] = 0

            # Round coefficients to integer
            zigzag_block = [int(round(coeff)) for coeff in zigzag_block]

            zigzag_blocks.append(zigzag_block)

    return zigzag_blocks
"""def process_image(image, array_H, transpose_H, zigzag_order, N):
    assert image.ndim == 3 and image.shape[2] == 3, "Image must be a color image with 3 color channels"
    zigzag_blocks = []
    height, width = image.shape[:2]

    # Process each 8x8 block of the image
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = image[i:i + 8, j:j + 8]

            # Apply the transformation to each color channel
            transformed_R = np.dot(np.dot(transpose_H, block[:, :, 0]), array_H)
            transformed_G = np.dot(np.dot(transpose_H, block[:, :, 1]), array_H)
            transformed_B = np.dot(np.dot(transpose_H, block[:, :, 2]), array_H)

            # Reorder the transformed block in a zigzag pattern
            zigzag_R = [transformed_R[tuple(element)] for element in zigzag_order]
            zigzag_G = [transformed_G[tuple(element)] for element in zigzag_order]
            zigzag_B = [transformed_B[tuple(element)] for element in zigzag_order]

            # Set the last N coefficients to 0
            for k in range(1, N + 1):
                zigzag_R[-k] = 0
                zigzag_G[-k] = 0
                zigzag_B[-k] = 0

            # Round coefficients to integer
            zigzag_R = [int(round(coeff)) for coeff in zigzag_R]
            zigzag_G = [int(round(coeff)) for coeff in zigzag_G]
            zigzag_B = [int(round(coeff)) for coeff in zigzag_B]

            # Combine the zigzag ordered channels into a single block
            zigzag_block = [(r,g,b) for r,g,b in zip(zigzag_R, zigzag_G, zigzag_B)]
            zigzag_blocks.append(zigzag_block)

    return zigzag_blocks"""

zigzag_blocks = process_image_grayscale(image, array_H, transpose_H, cikCakOrder, 5)
print("First Zigzag Block:", zigzag_blocks[0])
def generate_all_possible_triplets(range_max, M):
    all_triplets = [(i, i + 1, i + 2) for i in range(4, range_max - 2)]
    return sample(all_triplets, M)
def apply_F5_algorithm(zigzag_blocks, binary_message, N, M):
    seed(N * M)  # Set the seed for reproducibility
    range_max = min(32, 64 - N)

    for block in zigzag_blocks:
        selected_triplets = generate_all_possible_triplets(range_max, M)
        """print(selected_triplets)"""
        for triplet in selected_triplets:
            """print(triplet)"""
            AC1, AC2, AC3 = block[triplet[0]], block[triplet[1]], block[triplet[2]]
            C1, C2, C3 = AC1 & 1, AC2 & 1, AC3 & 1  # LSB
            # Izberemo dva bita iz sporočila
            x1, x2 = int(binary_message[0]), int(binary_message[1])
            binary_message = binary_message[2:]  # Odstranimo uporabljena bita iz sporočila

            # Operacije za skrivanje bitov
            if x1 != C1 and x2 == (C2 ^ C3):
                AC1 ^= 1  # Negacija LSB AC1
            elif x1 == C1 and x2 != (C2 ^ C3):
                AC3 ^= 1  # Negacija LSB AC3
            elif x1 != C1 and x2 != (C2 ^ C3):
                AC2 ^= 1  # Negacija LSB AC2

            # Nadomestimo spremenjene koeficiente nazaj v blok
            block[triplet[0]], block[triplet[1]], block[triplet[2]] = AC1, AC2, AC3

        if not binary_message:  # Če smo zaključili z vstavljanjem sporočila
            break

    return zigzag_blocks
zigzag_blocks_1 = apply_F5_algorithm(zigzag_blocks, binary_message, 5, 2)
print("First Modified Zigzag Block:", zigzag_blocks_1[0])  # Check the first block after modification
"""print(zigzag_blocks)"""

def run_length_encode(blocks):
    rle_blocks = []
    for block in blocks:
        rle_block = []
        count = 0
        for coeff in block:
            if coeff == 0:
                count += 1
            else:
                if count > 0:
                    rle_block.extend((0, count))  # Zapišemo teke ničel
                    count = 0
                rle_block.append(coeff)  # Zapišemo neničelni koeficient
        if count > 0:  # Zadnji tek ničel
            rle_block.extend((0, count))
        rle_blocks.append(rle_block)
    return rle_blocks
rle_blocks1 = run_length_encode(zigzag_blocks_1)
print("First RLE Block:", rle_blocks1[0])  # Check the first RLE block



def save_rle_to_binary(rle_blocks, filename):
    with open(filename, 'wb') as file:
        for block in rle_blocks:
            # Writing the length of the block first
            file.write(struct.pack('I', len(block)))
            for item in block:
                # Assuming each integer fits in 4 bytes
                file.write(struct.pack('i', item))

# Example usage
save_rle_to_binary(rle_blocks1, 'rle_data.bin')


"""DECODING"""

def read_rle_from_binary(filename):
    rle_blocks = []
    with open(filename, 'rb') as file:
        while True:
            length_bytes = file.read(4)
            if not length_bytes:
                break  # End of file
            length = struct.unpack('I', length_bytes)[0]
            block = []
            for _ in range(length):
                # Reading each integer (4 bytes as per your saving format)
                int_bytes = file.read(4)
                value = struct.unpack('i', int_bytes)[0]
                block.append(value)
            rle_blocks.append(block)
    return rle_blocks

# Example usage
rle_blocks2 = read_rle_from_binary('rle_data.bin')

def run_length_decode(rle_blocks):
    decoded_blocks = []
    for rle_block in rle_blocks:
        block = []
        i = 0
        while i < len(rle_block):
            coeff = rle_block[i]
            if coeff == 0:
                # Next element is the count of zeros
                count = rle_block[i + 1]
                block.extend([0] * count)  # Add 'count' number of zeros
                i += 2  # Skip over the count
            else:
                block.append(coeff)
                i += 1
        decoded_blocks.append(block)
    return decoded_blocks

# Example usage
zigzag_blocks_decoded = run_length_decode(rle_blocks1)
print("First Decoded Zigzag Block:", zigzag_blocks_decoded[0])  # Check the first decoded block


def extract_and_restore_F5(zigzag_blocks, N, M):
    seed(N * M)  # Set the seed for reproducibility, same as in embedding
    range_max = min(32, 64 - N)
    extracted_message = ''

    for block in zigzag_blocks:
        selected_triplets = generate_all_possible_triplets(range_max, M)
        for triplet in selected_triplets:
            AC1, AC2, AC3 = block[triplet[0]], block[triplet[1]], block[triplet[2]]
            C1, C2, C3 = AC1 & 1, AC2 & 1, AC3 & 1  # LSBs

            # Izračunamo ekstrahirana bita
            x1 = C1
            x2 = C2 ^ C3

            extracted_message += str(x1) + str(x2)

            # Restore original coefficients by reversing the embedding changes
            # This assumes that your embedding process is reversible
            # For example, if a coefficient was changed during embedding, you would reverse that change here
            block[triplet[0]], block[triplet[1]], block[triplet[2]] = AC1, AC2, AC3

            # Preverite, če je dovolj dolgo, da se ustavite (če je potrebno)
            # Na primer, če veste, koliko bitov je bilo vloženih ali če imate nekakšen konec sporočila

    return extracted_message, zigzag_blocks

# Example usage
extracted_binary_message, restored_zigzag_blocks = extract_and_restore_F5(zigzag_blocks, 5, 2)
if restored_zigzag_blocks == zigzag_blocks:
    print("True")

def debinarize_message(binary_message):
    # Extract the first 32 bits to find the message length
    message_length = int(binary_message[:32], 2)

    # Extract the actual message using the length
    binary_message = binary_message[32:32 + message_length]

    # Split the binary message into 8-bit chunks and convert back to characters
    message = ''.join(chr(int(binary_message[i:i+8], 2)) for i in range(0, len(binary_message), 8))

    return message

# Example usage
original_message = debinarize_message(binary_message)


"""do tuki je okkkk!!!!!!!!!!!!!!!"""
# till here it works fine
def inverse_process_image(zigzag_blocks, array_H, transpose_H, zigzag_order, N, height, width):
    assert len(zigzag_order) == 8*8, "Zigzag order must have 64 elements"
    reconstructed_image = np.zeros((height, width))

    # Process each block
    block_idx = 0
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            zigzag_block = zigzag_blocks[block_idx]
            block_idx += 1

            # Debug: Print the current block index and its first few values
            print(f"Block Index: {block_idx}, First few values: {zigzag_block[:5]}")

            # Reconstruct the original order of the block
            block = np.zeros((8, 8))
            for index, element in enumerate(zigzag_order):
                block[element] = zigzag_block[index]

            # Debug: Print the reconstructed block (first few values)
            print(f"Reconstructed block (first few values): {block.ravel()[:5]}")

            # Inverse transformation
            reconstructed_block = np.dot(np.dot(array_H, block), transpose_H)

            # Debug: Print min, max of the reconstructed block
            print(f"Reconstructed block min, max: {reconstructed_block.min()}, {reconstructed_block.max()}")

            # Assign to the corresponding position in the image
            reconstructed_image[i:i + 8, j:j + 8] = reconstructed_block

            # Debug: Print min, max of the overall reconstructed image so far
            print(f"Reconstructed image min, max so far: {reconstructed_image.min()}, {reconstructed_image.max()}")

    return reconstructed_image



# Example usage
# Assume height and width are known or stored
restored_image = inverse_process_image(restored_zigzag_blocks, array_H, transpose_H, cikCakOrder, 5, height, width)

print(restored_image.min(), restored_image.max())
print(image.min(), image.max())

def scale_image(restored_image, target_min, target_max):
    min_val, max_val = image.min(), image.max()
    scaled_image = (image - min_val) / (max_val - min_val) * (target_max - target_min) + target_min
    return scaled_image

scaled_image = scale_image(restored_image, 0, 255)
clipped_image = np.clip(scaled_image, 0, 255).astype(np.uint8)
print(clipped_image.min(), clipped_image.max())

import imageio.v2

def unpad_image(padded_image, original_height, original_width):
    """
    Remove padding from the image.
    Args:
    padded_image (numpy.ndarray): The padded image.
    original_height (int): The original height of the image before padding.
    original_width (int): The original width of the image before padding.

    Returns:
    numpy.ndarray: The unpadded image.
    """
    # Remove the padding
    unpadded_image = padded_image[:original_height, :original_width]

    return unpadded_image


import numpy as np
import imageio.v2

def save_image(image, path):
    """
    Save the image to a file, converting it to 8-bit format if necessary.
    Args:
    image (numpy.ndarray): The image to save.
    path (str): The file path to save the image.
    """

    # Check if the image is in floating-point format and convert to 8-bit grayscale
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Clip the pixel values to the range [0, 1]
        clipped_image = np.clip(image, 0, 1)

        # Scale the pixel values to 0-255
        image_8bit = (clipped_image * 255).astype(np.uint8)
    else:
        image_8bit = image.astype(np.uint8)

    # Save the image
    imageio.v2.imwrite(path, image_8bit)

# Example usage
restored_image_path = 'restored_image.bmp'  # Path to save the restored image

original_height, original_width =  height, width  # Assume these are known or stored
unpadded_image = unpad_image(clipped_image, original_height, original_width)


save_image(unpadded_image, restored_image_path)
print(original_message)





