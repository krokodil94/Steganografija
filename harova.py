import numpy as np
import math
import imageio
import heapq
import sys
from collections import defaultdict
import json
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
    """
    Reads an image from the given file path and pads it to ensure its dimensions are multiples of 8.

    The function performs the following steps:
    1. Reads the image from the specified path.
    2. Calculates the original height and width of the image.
    3. Determines the necessary padding to make the height and width multiples of 8.
       This is important for processing the image in 8x8 blocks.
    4. Applies the calculated padding to the image. The padding is added to the bottom
       and right sides of the image, filling with zeros (black pixels).

    Input:
    - image_path: A string representing the path to the image file.

    Output:
    - padded_image: The padded image as a NumPy array.
    - height: The original height of the image (before padding).
    - width: The original width of the image (before padding).
    """
    # Read the image from the given path
    image = imageio.v2.imread(image_path)

    # Extract the original height and width of the image
    height, width = image.shape[:2]

    # Calculate the padding required to make height and width multiples of 8
    height_padding = (8 - height % 8) % 8
    width_padding = (8 - width % 8) % 8

    # Apply padding to the image. Padding is added to the bottom and right sides.
    # The padding is filled with zeros (black pixels).
    padded_image = np.pad(image,
                          ((0, height_padding), (0, width_padding), (0, 0)),
                          mode='constant',
                          constant_values=0)

    # Return the padded image and its original dimensions
    return padded_image, height, width


def process_image(image, array_H, transpose_H, zigzag_order):
    """
    Processes the image by dividing it into 8x8 blocks and applying a transformation followed by zigzag reordering.

    Steps:
    1. Iterates through the image in 8x8 blocks.
    2. For each block, applies a transformation using the provided matrices (array_H and transpose_H).
       This transformation is done separately for each color channel (Red, Green, Blue).
    3. After the transformation, reorders the elements of each block in a zigzag pattern.
       This is mainly used in JPEG compression to arrange the coefficients in a specific order.

    Input:
    - image: A NumPy array representing the image to be processed.
    - array_H: The transformation matrix.
    - transpose_H: The transpose of the transformation matrix.
    - zigzag_order: A list of tuples representing the zigzag order for traversing the 8x8 blocks.

    Output:
    - zigzag_blocks: A list of tuples representing the transformed and zigzag reordered blocks of the image.
                     Each tuple contains the transformed values for the Red, Green, and Blue channels.
    """
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

            # Combine the zigzag ordered channels into a single block
            zigzag_block = [(r, g, b) for r, g, b in zip(zigzag_R, zigzag_G, zigzag_B)]
            zigzag_blocks.extend(zigzag_block)

    return zigzag_blocks


def treshold(zigzag_blocks, T):
    """
        Applies a threshold to the zigzag blocks of an image.

        This function processes each block of transformed and zigzag reordered image data.
        It sets any value in the block that is below a certain threshold (T) to zero.

        Input:
        - zigzag_blocks: A list of tuples representing the transformed and zigzag reordered blocks of the image.
                         Each tuple contains values for the Red, Green, and Blue channels.
        - T: The threshold value. Any value in the block with an absolute value less than T is set to zero.

        Output:
        - A list of tuples similar to the input, but with values below the threshold set to zero.
        """
    return [tuple(0 if abs(value) < T else value for value in block) for block in zigzag_blocks]

def generate_huffman_tree(freq):
    """
    Generates a Huffman tree for encoding based on the frequencies of values.

    Input:
    - freq: A dictionary where keys are values to be encoded (e.g., pixel values) and values are their frequencies.

    Output:
    - A list representing the Huffman tree. Each element is a list containing the symbol and its binary code as a string.
    """
    # Create a priority queue (min-heap) based on the frequencies
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)

    # Build the Huffman tree
    while len(heap) > 1:
        # Pop two elements with the smallest frequencies
        first, second = heapq.heappop(heap), heapq.heappop(heap)

        # Append '0' to the codes of symbols in the first element and '1' to those in the second
        for pair in first[1:]:
            pair[1] = '0' + pair[1]
        for pair in second[1:]:
            pair[1] = '1' + pair[1]

        # Combine the two elements into a new node and push it back into the heap
        new_freq = first[0] + second[0]
        new_item = [new_freq] + first[1:] + second[1:]
        heapq.heappush(heap, new_item)

    # Return the Huffman tree
    return heap[0][1:]


def huffman_encode(compressed):
    """
    Encodes a list of values using Huffman coding.

    This function first calculates the frequency of each value in the input list.
    It then uses these frequencies to generate a Huffman tree, which is a binary tree
    where each leaf node represents a value from the input list and its associated binary code.
    The function finally goes through the input list again and replaces each value with its binary code
    as determined by the Huffman tree.

    Input:
    - compressed: A list of values (e.g., from a compressed image) to be encoded using Huffman coding.

    Output:
    - freq: A dictionary where the keys are the values from the input list and the values are their frequencies.
    - encoded_values: A list of binary codes corresponding to the input values based on the Huffman tree.
    - huffman_codes: A dictionary mapping each value to its corresponding binary code.
    """
    # Calculate the frequency of each value in the input list
    freq = defaultdict(int)
    for value in compressed:
        freq[value] += 1

    # Generate the Huffman tree using these frequencies
    huffman_tree = generate_huffman_tree(freq)

    # Create a dictionary mapping each value to its Huffman code
    huffman_codes = {value: code for value, code in huffman_tree}

    # Replace each value in the input list with its corresponding Huffman code
    encoded_values = [int(bit) for value in compressed for bit in huffman_codes[value]]

    return freq, encoded_values, huffman_codes



def convert_to_binary_data(encoded_bits):
    """
    Converts a list of encoded bits into a binary data format suitable for storage or transmission.

    This function performs the following steps:
    1. Converts the list of bits into a single string of binary digits.
    2. Adds padding to this string to make its length a multiple of 8, as bytes consist of 8 bits.
       This is necessary for converting the string into bytes.
    3. Converts the padded binary string into a bytes object, which is a sequence of bytes that can be stored or transmitted.

    Input:
    - encoded_bits: A list of bits (0s and 1s) representing encoded data.

    Output:
    - binary_bytes: A bytes object representing the encoded data.
    - padding_size: The number of padding bits added to make the length a multiple of 8.
    """
    # Convert the list of bits into a binary string
    binary_str = ''.join(map(str, encoded_bits))

    # Calculate the padding size needed to make the binary string's length a multiple of 8
    padding_size = (8 - len(binary_str) % 8) % 8

    # Add the necessary padding (0s) to the binary string
    padded_binary_str = binary_str + '0' * padding_size

    # Convert the padded binary string into a bytes object
    binary_bytes = int(padded_binary_str, 2).to_bytes((len(padded_binary_str) + 7) // 8, 'big')

    return binary_bytes, padding_size



def save_to_binary_file(filename, binary_data):
    with open(filename, 'wb') as file:
        file.write(binary_data)

"""DECODING"""
def read_binary_file(filename):
    with open(filename, 'rb') as file:
        binary_data = file.read()
    return binary_data

def convert_from_binary(binary_data, padding):
    """
    Converts binary data back into a list of encoded bits.

    This function reverses the process of the 'convert_to_binary_data' function. It performs the following steps:
    1. Converts each byte in the binary data into a string of 8 binary digits (bits),
       effectively reconstructing the original binary string before it was converted into bytes.
    2. Removes the padding at the end of this binary string that was added during the encoding process.
    3. Converts the binary string back into a list of individual bits.

    Input:
    - binary_data: The binary data (bytes object) to be converted.
    - padding: The number of padding bits that were added to make the binary string's length a multiple of 8 during encoding.

    Output:
    - encoded_values: A list of bits (0s and 1s) representing the original encoded data.
    """
    # Convert the binary data (bytes) into a single binary string
    binary_string = ''.join(format(byte, '08b') for byte in binary_data)

    # Remove the padding added during the encoding process
    binary_string = binary_string[:-padding] if padding > 0 else binary_string

    # Convert the binary string back into a list of individual bits
    encoded_values = [int(bit) for bit in binary_string]

    return encoded_values


def build_decoding_tree(huffman_codes):
    """
    Builds a decoding tree from Huffman codes.

    This function constructs a binary tree that can be used to decode Huffman-encoded data. Each path from
    the root of the tree to a leaf node represents a Huffman code, and the leaf node itself represents the
    corresponding character (or value).

    Input:
    - huffman_codes: A dictionary where keys are characters (or values) and values are their corresponding
                     Huffman codes as strings of binary digits.

    Output:
    - root: The root node of the binary decoding tree. This tree is a nested dictionary where each key is
            a bit ('0' or '1') and each value is the next node in the path. Leaf nodes have an additional key 'char'
            indicating the character (or value) that the path represents.
    """
    root = {}  # Initialize the root of the tree

    # Build the tree by adding paths for each Huffman code
    for char, code in huffman_codes.items():
        node = root
        for bit in code:
            # For each bit in the Huffman code, traverse the tree and create new nodes as needed
            node = node.setdefault(bit, {})
        # Set the leaf node to represent the character
        node['char'] = char

    return root


def huffman_decode(values, huffman_codes):
    """
    Decodes a list of values using the provided Huffman codes.

    This function uses a Huffman decoding tree to decode a sequence of binary values (0s and 1s)
    back into their original values (e.g., pixel values in an image). The decoding tree is built
    from the provided Huffman codes.

    Input:
    - values: A list of binary values (0s and 1s) representing Huffman encoded data.
    - huffman_codes: A dictionary where keys are the original values and values are their corresponding Huffman codes.

    Output:
    - decoded_values: A list of the original values decoded from the input binary sequence.
    """
    # Build the decoding tree from the Huffman codes
    tree = build_decoding_tree(huffman_codes)

    decoded_values = []  # List to hold the decoded values
    node = tree  # Start from the root of the decoding tree

    # Iterate through each binary value and traverse the decoding tree
    for value in values:
        # Move to the next node in the tree based on the binary value
        node = node[str(value)]

        # Check if the current node is a leaf node (i.e., end of a Huffman code)
        if 'char' in node:
            # Append the decoded value and reset to start from the root again
            decoded_values.append(node['char'])
            node = tree

    return decoded_values


def inverse_process_image(zigzag_blocks, array_H, transpose_H, zigzag_order, width, height):
    """
    Reconstructs an image from its zigzag-ordered blocks.

    This function reverses the transformation and zigzag ordering applied during the image compression process.
    It reconstructs the image block by block, each of which is 8x8 pixels.

    Input:
    - zigzag_blocks: A list of tuples representing the transformed and zigzag reordered blocks of the image.
    - array_H: The transformation matrix used in the forward transformation.
    - transpose_H: The transpose of the transformation matrix.
    - zigzag_order: A list of tuples representing the zigzag order for traversing the 8x8 blocks.
    - width: The original width of the image (before padding).
    - height: The original height of the image (before padding).

    Output:
    - inverse_image: The reconstructed image as a NumPy array.
    """
    # Initialize the reconstructed image array
    inverse_image = np.zeros((height, width, 3), dtype=np.float32)

    block_size = 8  # Size of each block (8x8 pixels)
    tuples_per_block = block_size * block_size  # Number of elements per block

    # Process each block
    for block_num, block_start in enumerate(range(0, len(zigzag_blocks), tuples_per_block)):
        # Initialize an empty block
        inverse_block = np.zeros((block_size, block_size, 3), dtype=np.float32)

        # Fill the block with values from zigzag_blocks in zigzag order
        for i, (x, y) in enumerate(zigzag_order):
            inverse_block[x, y] = zigzag_blocks[block_start + i]

        # Apply the inverse transformation to each color channel
        for channel in range(3):
            inverse_block[:, :, channel] = np.dot(np.dot(array_H, inverse_block[:, :, channel]), transpose_H)

        # Determine the position of the block in the image
        row = (block_num // (width // block_size)) * block_size
        col = (block_num % (width // block_size)) * block_size
        block_height = min(block_size, height - row)
        block_width = min(block_size, width - col)

        # Place the processed block into the reconstructed image
        inverse_image[row:row + block_height, col:col + block_width] = inverse_block[:block_height, :block_width]

    return inverse_image

def parse_arguments():
    input_file = sys.argv[1]
    option = sys.argv[2]
    output_file = sys.argv[3]
    threshold = float(sys.argv[4]) if len(sys.argv) == 5 else None

    return input_file, option, output_file, threshold

def main():
    """
    Main function to execute the image compression and decompression process.

    This function reads command line arguments to determine the operation mode (compression or decompression),
    input file, output file, and threshold for compression. Based on the operation mode, it either compresses
    an image and saves it along with its metadata or decompresses an image using the stored metadata.
    """
    # Parse command line arguments
    input_file, option, output_file, threshold = parse_arguments()

    if option == 'c':
        # Compression mode
        # Read and pad the input image
        padded_image, height, width = read_image(input_file)

        # Process the image to get zigzag ordered blocks
        zigzag_blocks = process_image(padded_image, array_H, transpose_H, cikCakOrder)

        # Apply threshold to the zigzag blocks
        compressed = threshold(zigzag_blocks, threshold)

        # Encode the compressed blocks using Huffman coding
        freq, encoded_values, huffman_codes = huffman_encode(compressed)

        # Convert the encoded values to binary data
        binary_data, padding = convert_to_binary_data(encoded_values)

        # Save the binary data to an output file
        save_to_binary_file(output_file, binary_data)
        print(f"Slika je bila uspešno shranjena v datoteko {output_file}.")

        # Save metadata required for decompression
        huffman_codes_str = {str(key): value for key, value in huffman_codes.items()}
        metadata = {
            "padding": padding,
            "huffman_codes": huffman_codes_str,
            "width": width,
            "height": height
        }
        with open(output_file + '.metadata', 'w') as metafile:
            json.dump(metadata, metafile)

    elif option == 'd':
        # Decompression mode
        # Read the metadata file
        with open(input_file + '.metadata', 'r') as metafile:
            metadata = json.load(metafile)

        # Extract metadata information
        padding = metadata["padding"]
        huffman_codes_str = metadata["huffman_codes"]
        width = metadata["width"]
        height = metadata["height"]

        # Convert string-based Huffman codes to their original form
        huffman_codes = {eval(key): value for key, value in huffman_codes_str.items()}

        # Read the binary data from the input file
        binary_data = read_binary_file(input_file)

        # Convert binary data back to encoded values
        values = convert_from_binary(binary_data, padding)

        # Decode the values using Huffman decoding
        decoded_values = huffman_decode(values, huffman_codes)

        # Reconstruct the image from the decoded values
        inverse_image = inverse_process_image(decoded_values, array_H, transpose_H, cikCakOrder, width, height)

        # Save the reconstructed image
        imageio.imwrite(output_file, inverse_image.astype(np.uint8))
        print(f"Slika je shranjena v datoteko {output_file}.")

if __name__ == '__main__':
    main()


