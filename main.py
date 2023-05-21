import cv2
import numpy as np

def calculate_homogeneity(image):
    # convertir l'image en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # calculer la déviation standard et la moyenne des pixels de l'image
    std_dev = np.std(gray_image) #calculer l'ecart-type
    mean = np.mean(gray_image) #calculer la moyenne

    # calculer l'homogénéité
    homogeneity = std_dev / mean

    return homogeneity

def find_best_image(image_paths):
    best_homogeneity = 0.0
    best_image = None

    for path in image_paths:
        # charger l'image
        image = cv2.imread(path)

        # calculer l'homogénéité de l'image
        homogeneity = calculate_homogeneity(image)

        # mettre à jour la meilleure homogénéité et l'image correspondante
        if homogeneity > best_homogeneity:
            best_homogeneity = homogeneity
            best_image = image

    return best_image, best_homogeneity

# liste des chemins d'accès aux images à comparer
image_paths = ['messi.jpg','ibiza.jpg','compus.jpg','cooper.jpg','aveo.jpg']

# trouver l'image avec la meilleure homogénéité
best_image, best_homogeneity = find_best_image(image_paths)

# afficher l'image avec la meilleure homogénéité
cv2.imshow("Best Image", best_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# afficher la meilleure homogénéité
print("Best Homogeneity:", best_homogeneity)