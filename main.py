import numpy as np
import cv2


# def image_to_matrix(image_path):
#     # Read the image using OpenCV
#     image = cv2.imread(image_path)
#
#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Convert the grayscale image to a matrix
#     matrix = np.array(gray_image)
#
#     return matrix
#
#
# # Provide the path to your image file
# image_path = 'nissan.jpg'
#
# # Convert the image to a matrix
# matrix = image_to_matrix(image_path)
#
# # Print the matrix
# print(matrix)
# print("-------------------------------------")

# Given matrix
matrix = np.array([[121, 71, 145, 175, 178, 180],
                   [135, 230, 255, 255, 179, 181],
                   [18, 244, 250, 255, 181, 183],
                   [121, 244, 250, 0, 178, 180],
                   [135, 81, 87, 176, 179, 181],
                   [18, 11, 181, 179, 181, 183],
                   [164, 161, 157, 221, 229, 234],
                   [160, 157, 155, 215, 222, 228],
                   [157, 155, 154, 212, 216, 221]])

# Define positions of submatrices
positions = [(1, 1), (0, 1), (0, 0), (1, 0)]
correlation_results = []



# Extract submatrix at the first position
submatrix_1 = matrix[positions[0][0]:positions[0][0]+3, positions[0][1]:positions[0][1]+3]
linear_submatrix_1 = submatrix_1.flatten()
sorted_submatrix_1 = np.sort(linear_submatrix_1)[::1]

print(f"Submatrix at position {positions[0]}:")
print(sorted_submatrix_1)
print()

# Calculate correlations between the first position and other positions
for i in range(1, len(positions)):
    row, column = positions[i]
    submatrix_i = matrix[row:row+3, column:column+3]
    linear_submatrix_i = submatrix_i.flatten()
    sorted_submatrix_i = np.sort(linear_submatrix_i)[::1]

    # Calculate correlation
    correlation = np.corrcoef(linear_submatrix_1, sorted_submatrix_i)[0, 1]

    print(f"Correlation between {positions[0]} and {positions[i]}:")
    print(correlation)
    print()