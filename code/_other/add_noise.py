import cv2
import numpy as np
import os

# Функція для застосування гауссівського шуму до зображення
def apply_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss, 0, 255)
    return noisy.astype(np.uint8)

# Шлях до теки з оригінальними зображеннями
input_folder = 'Coursework\people'

# Шлях до теки для збереження погіршених зображень
output_folder = 'Coursework\people_after_noise'
os.makedirs(output_folder, exist_ok=True)

# Зчитуємо всі файли зображень з вхідної теки
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Перебираємо всі файли зображень
for image_file in image_files:
    # Зчитуємо оригінальне зображення
    original_image = cv2.imread(os.path.join(input_folder, image_file))

    # Застосовуємо гауссівський шум для погіршення якості
    noisy_image = apply_gaussian_noise(original_image)

    # Зберігаємо погіршене зображення у вихідну теку
    cv2.imwrite(os.path.join(output_folder, f'noisy_{image_file}'), noisy_image)

print("Погіршення якості зображень завершено.")
