import numpy as np
import cv2

# Provide the path to your image file
image_path = 'images/cooper.jpg'

def image_to_matrix(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to a matrix
    matrix = np.array(gray_image)

    return matrix





# Convert the image to a matrix
matrix = image_to_matrix(image_path)

# Print the matrix
print(matrix)
print("-------------------------------------")

# matrix = np.array([[121, 71, 145, 175, 178, 180, 175, 178, 180],
#                    [135, 230, 255, 255, 179, 181, 175, 178, 180],
#                    [18, 244, 250, 255, 181, 183, 175, 178, 180],
#                    [121, 244, 250, 0, 178, 180, 175, 178, 180],
#                    [135, 81, 87, 176, 179, 181, 175, 178, 180],
#                    [18, 11, 181, 179, 181, 183, 175, 178, 180],
#                    [164, 161, 157, 221, 229, 234, 175, 178, 180],
#                    [160, 157, 155, 215, 222, 228, 175, 178, 180],
#                    [157, 155, 154, 212, 216, 221, 175, 178, 180]])

num_rows = matrix.shape[0]
num_columns = matrix.shape[1]

new_matrix = np.zeros((num_rows - 2, num_columns - 2), dtype=int)  # Initialize the new matrix

# Perform the operation on the matrix
for row in range(2, num_rows - 1):
    for column in range(2, num_columns - 1):
        # Get the submatrix
        submatrix = matrix[row - 1:row + 2, column - 1:column + 2]


        #Print the submatrix
        print(f"Submatrix at ({row}, {column}):")
        print(submatrix)

        print("Top:", matrix[row - 2:row + 1, column-1:column + 2])
        print("Left:", matrix[row -1:row + 2, column - 2:column + 1])
        print("Top-Left:", matrix[row - 2:row + 1, column-2:column + 1])
        print("-------------------------------------")

        # Check if the submatrix contains valid values
        if np.isnan(submatrix).any() or not np.isfinite(submatrix).all():
            new_matrix[row - 2, column - 2] = matrix[row, column]
        else:
            # Compute the correlations
            top_correlation = np.corrcoef(submatrix.ravel(), matrix[row - 2:row + 1, column-1:column + 2].ravel())
            left_correlation = np.corrcoef(submatrix.ravel(), matrix[row -1:row + 2, column - 2:column + 1].ravel())
            top_left_correlation = np.corrcoef(submatrix.ravel(), matrix[row - 2:row + 1, column-2:column + 1].ravel())

            # Print the correlations
            print("Top correlation:", top_correlation[0, 1])
            print("Left correlation:", left_correlation[0, 1])
            print("Top-left correlation:", top_left_correlation[0, 1])
            print("-------------------------------------")

            # Compute the average correlation
            average_correlation = (top_correlation[0, 1] + left_correlation[0, 1] + top_left_correlation[0, 1]) / 3
            print("average correlation:", average_correlation)

            # Compute the average of the values
            average_value = int(np.mean(submatrix))
            print("average value:", average_value)
            print("-------------------------------------")

            if average_correlation > 0.12 or average_correlation < -0.12:
                new_matrix[row - 2, column - 2] = average_value
            else:
                new_matrix[row - 2, column - 2] = matrix[row, column]

print(new_matrix)

# Convert the new matrix back to an image
new_image = new_matrix.astype(np.uint8)

# Display the image
cv2.imshow('New Image', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()