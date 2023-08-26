import os
import cv2
import random
import numpy as np


def adjust_hue(image, hue_factor):
    """
      Adjust the hue of an image
      :param image: Input image
      :param hue_factor: Hue adjustment factor, 0 means no adjustment,
                        negative values rotate counterclockwise, positive values rotate clockwise
      :return: Adjusted image
      """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[..., 0] = (hsv_image[..., 0] + hue_factor) % 180
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return adjusted_image

def adjust_brightness(image, beta):
    """
        Adjust the brightness of an image (illumination)
        :param image: Input image
        :param beta: Brightness adjustment factor, 0 means no adjustment,
                     positive values increase brightness, negative values decrease brightness
        :return: Adjusted image
        """
    adjusted_image = cv2.convertScaleAbs(image, alpha=1, beta=beta)
    return adjusted_image


def adjust_saturation(image, saturation_factor):
    """
        Adjust the saturation of an image
        :param image: Input image
        :param saturation_factor: Saturation adjustment factor, 1 means no adjustment,
                                 less than 1 decreases saturation, greater than 1 increases saturation
        :return: Adjusted image
        """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[..., 1] = hsv_image[..., 1] * saturation_factor
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return adjusted_image


def add_gaussian_noise(image, mean=0, std=25):
    """
        Add Gaussian noise to an image.
        :param image: Input image.
        :param mean: Mean of the noise.
        :param std: Standard deviation of the noise.
        :return: Noisy image.
        """
    h, w, c = image.shape
    noise = np.random.normal(mean, std, (h, w, c))
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def adjust_contrast(image, alpha):
    """
    Adjust the contrast of an image
    :param image: Input image
    :param alpha: Contrast adjustment factor, 1 means no adjustment,
                  less than 1 decreases contrast, greater than 1 increases contrast
    :return: Adjusted image
    """
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return adjusted_image


def mosaic(img, block_size):
    h, w = img.shape[:2]
    block_w = w // block_size
    block_h = h // block_size

    for y in range(block_h):
        for x in range(block_w):
            roi = img[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size]
            mean_color = roi.mean(axis=(0, 1), dtype=int)
            img[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size] = mean_color

    return img

def apply_blur(image, blur_factor):
    """
    Apply Gaussian blur to an image
    :param image: Input image
    :param blur_factor: Blurring strength, larger values result in more noticeable blur
    :return: Blurred image
    """
    adjusted_image = cv2.GaussianBlur(image, (blur_factor, blur_factor), 0)
    return adjusted_image

def translate_image(image, dx, dy):
    """
    Translate an image
    :param image: Input image
    :param dx: Translation amount in the x direction
    :param dy: Translation amount in the y direction
    :return: Translated image
    """
    rows, cols, _ = image.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    adjusted_image = cv2.warpAffine(image, M, (cols, rows))
    return adjusted_image

def shear_image(image, shear_factor_x, shear_factor_y):
    """
    Shear an image
    :param image: Input image
    :param shear_factor_x: Shear factor in the x direction
    :param shear_factor_y: Shear factor in the y direction
    :return: Sheared image
    """
    rows, cols, _ = image.shape
    M = np.float32([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]])
    adjusted_image = cv2.warpAffine(image, M, (cols, rows))
    return adjusted_image

def crop_object(image, x, y, width, height):
    adjusted_image = image[y:y+height, x:x+width]
    return adjusted_image


def rotate_image(image, angle):
    """
    Rotate an image
    :param image: Input image
    :param angle: Rotation angle, positive values for counterclockwise rotation, negative values for clockwise rotation
    :return: Rotated image
    """
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    adjusted_image = cv2.warpAffine(image, M, (cols, rows))
    return adjusted_image


def resize_objects_center(img, scale_factor):
    """
    Scale an image
    :param image: Input image
    :param scale_factor: Scaling factor, greater than 1 for enlargement, less than 1 for reduction
    :return: Scaled image
    """
    h, w = img.shape[:2]
    scaled_h = int(h * scale_factor)
    scaled_w = int(w * scale_factor)

    scaled_img = cv2.resize(img, (scaled_w, scaled_h))

    top = (h - scaled_h) // 2
    bottom = h - scaled_h - top
    left = (w - scaled_w) // 2
    right = w - scaled_w - left

    adjusted_image = cv2.copyMakeBorder(scaled_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return adjusted_image


def flip_image(image, flip_code):
    adjusted_image = cv2.flip(image, flip_code)
    return adjusted_image


def add_random_occlusion(image, num_occlusions):
    """
    Add random occlusions to an image.
    :param image: Input image.
    :param num_occlusions: Number of occlusions to add.
    :return: Image with occlusions.
    """
    h, w, _ = image.shape

    for _ in range(num_occlusions):
        occlusion_height = np.random.randint(10, h // 3)  # Occlusion height range
        occlusion_width = np.random.randint(10, w // 3)  # Occlusion width range

        # Randomly generate occlusion region position
        x1 = np.random.randint(0, w - occlusion_width)
        y1 = np.random.randint(0, h - occlusion_height)
        x2 = x1 + occlusion_width
        y2 = y1 + occlusion_height

        # Set occlusion region pixels to a random value (color or other values)
        random_color = np.random.randint(0, 256, (1, 1, 3), dtype=np.uint8)
        image[y1:y2, x1:x2] = random_color

    return image

def cutout_image(image, num_patches, max_patch_size):
    """
    Apply Cutout augmentation to an image.
    :param image: Input image.
    :param num_patches: Number of patches to cut out.
    :param max_patch_size: Maximum size of each patch.
    :return: Image with Cutout applied.
    """
    h, w, _ = image.shape

    for _ in range(num_patches):
        patch_size = random.randint(1, max_patch_size)
        x1, y1 = random.randint(0, w - patch_size), random.randint(0, h - patch_size)
        x2, y2 = x1 + patch_size, y1 + patch_size

        image[y1:y2, x1:x2] = 0

    return image

def cutmix_images(image1, image2, alpha):
    h, w, _ = image1.shape
    x1, y1 = random.randint(0, w), random.randint(0, h)
    x2, y2 = random.randint(0, w), random.randint(0, h)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    mixed_image = image1.copy()
    mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]

    return mixed_image

def mixup_images(image1, image2, alpha):
    adjusted_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    return adjusted_image


# Get a list of image filenames from the specified folder
path = "input_images"
img_files = os.listdir(path)
images = []
for img_file in img_files:
    if img_file.endswith('.jpg') or img_file.endswith('.png'):
        input_path = os.path.join(path, img_file)
        img = cv2.imread(input_path)
        if img is not None:
            images.append(img)

# This function processes a set of images based on selected operations. It applies a series of operations to each image and saves the processed results to an output folder.
def process_images(input_folder, output_folder, operations):
    # Get a list of image files in the input folder
    img_files = os.listdir(input_folder)

    for img_file in img_files:
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            # Construct the paths for input and output images
            input_path = os.path.join(input_folder, img_file)
            output_path = os.path.join(output_folder, img_file)
            # Read the image using OpenCV
            img = cv2.imread(input_path)
            # Create a copy of the image for processing
            processed_img = img.copy()
            # Initialize a list with the original image
            adjusted_images = [processed_img.copy()]  # Initialize with the original image
            # Loop through selected operations and apply them to each image
            for operation in operations:
                new_adjusted_images = []

                if operation == 1:
                    # Adjust hue with different factors
                    hue_factors = [30, 10, -30]
                    for hue_factor in hue_factors:
                        for adjusted_img in adjusted_images:
                            new_adjusted_img = adjust_hue(adjusted_img.copy(), hue_factor)
                            new_adjusted_images.append(new_adjusted_img)

                if operation == 2:
                    # Adjust brightness with different factors
                    brightness_factors = [100, 50]
                    for brightness_factor in brightness_factors:
                        for adjusted_img in adjusted_images:
                            new_adjusted_img = adjust_brightness(adjusted_img.copy(), brightness_factor)
                            new_adjusted_images.append(new_adjusted_img)

                if operation == 3:
                    # Adjust saturation with different factors
                    saturation_factors = [1.5, 1, 0.5, 0]
                    for saturation_factor in saturation_factors:
                        for adjusted_img in adjusted_images:
                            new_adjusted_img = adjust_saturation(adjusted_img.copy(), saturation_factor)
                            new_adjusted_images.append(new_adjusted_img)

                if operation == 4:
                    # Add Gaussian noise with different levels
                    noise_levels = [25, 50, 70, 90, 110]
                    for noise_level in noise_levels:
                        for adjusted_img in adjusted_images:
                            noisy_img = add_gaussian_noise(adjusted_img.copy(), mean=0, std=noise_level)
                            new_adjusted_images.append(noisy_img)

                if operation == 5:
                    # Adjust contrast with different factors
                    contrast_factors = [0.5, 1.5]
                    for contrast_factor in contrast_factors:
                        for adjusted_img in adjusted_images:
                            new_adjusted_img = adjust_contrast(adjusted_img.copy(), contrast_factor)
                            new_adjusted_images.append(new_adjusted_img)

                if operation == 6:
                    # Apply mosaic effect with different block sizes
                    block_sizes = [1, 4, 6, 10]
                    for block_size in block_sizes:
                        for adjusted_img in adjusted_images:
                            new_adjusted_img = mosaic(adjusted_img.copy(), block_size)
                            new_adjusted_images.append(new_adjusted_img)

                if operation == 7:
                    # Apply blur with different factors
                    blur_factors = [25, 55, 75]
                    for blur_factor in blur_factors:
                        for adjusted_img in adjusted_images:
                            new_adjusted_img = apply_blur(adjusted_img.copy(), blur_factor)
                            new_adjusted_images.append(new_adjusted_img)


                if operation == 8:
                    # Apply translation to the images with different offsets
                    translation_offsets = [(50, 50), (100, 100)]
                    for dx, dy in translation_offsets:
                        for adjusted_img in adjusted_images:
                            new_adjusted_img = translate_image(adjusted_img.copy(), dx, dy)
                            new_adjusted_images.append(new_adjusted_img)


                if operation == 9:
                    # Apply shearing to the images with different factors
                    shear_factors = [(0.2, 0.1), (-0.1, 0.3), (0.4, -0.4)]
                    for shear_factor_x, shear_factor_y in shear_factors:
                        for adjusted_img in adjusted_images:
                            new_adjusted_img = shear_image(adjusted_img.copy(), shear_factor_x=shear_factor_x, shear_factor_y=shear_factor_y)
                            new_adjusted_images.append(new_adjusted_img)

                if operation == 10:
                    # Crop objects from images with specified coordinates
                    object_coords = [(200, 150, 400, 400)]
                    for idx, (object_x, object_y, object_width, object_height) in enumerate(object_coords):
                        for adjusted_img in adjusted_images:
                            new_adjusted_img = crop_object(adjusted_img.copy(), object_x, object_y, object_width, object_height)
                            new_adjusted_images.append(new_adjusted_img)

                if operation == 11:
                    rotation_angles = [30, -15, 60]
                    for rotation_angle in rotation_angles:
                        for adjusted_img in adjusted_images:
                            new_adjusted_img = rotate_image(adjusted_img.copy(), rotation_angle)
                            new_adjusted_images.append(new_adjusted_img)

                if operation == 12:
                    scale_factors = [0.5, 0.3]
                    for scale_factor in scale_factors:
                        for adjusted_img in adjusted_images:
                            new_adjusted_img = resize_objects_center(adjusted_img.copy(), scale_factor)
                            new_adjusted_images.append(new_adjusted_img)

                if operation == 13:
                    flip_codes = [1, 0, -1]
                    for flip_code in flip_codes:
                        for adjusted_img in adjusted_images:
                            new_adjusted_img = flip_image(adjusted_img.copy(), flip_code)
                            new_adjusted_images.append(new_adjusted_img)

                if operation == 14:
                    num_patches = 3
                    max_patch_size = 50
                    for adjusted_img in adjusted_images:
                        new_adjusted_img = cutout_image(adjusted_img.copy(), num_patches,max_patch_size )
                        new_adjusted_images.append(new_adjusted_img)

                if operation == 15:
                    # Splice images together with resizing
                    selected_images = random.sample(images, min(2, len(images)))

                    while len(selected_images) < 2:
                        selected_images.append(random.choice(images))

                    image1 = selected_images[0]
                    image2 = selected_images[1]

                    min_height = min(image1.shape[0], image2.shape[0])
                    image1_resized = cv2.resize(image1,
                                                (int(image1.shape[1] * min_height / image1.shape[0]), min_height))
                    image2_resized = cv2.resize(image2,
                                                (int(image2.shape[1] * min_height / image2.shape[0]), min_height))
                    concatenated_image = np.concatenate((image1_resized, image2_resized), axis=1)
                    new_adjusted_images.append(concatenated_image)

                if operation == 16:
                    # Add random occlusions to the images
                    num_occlusions_list = 2
                    for adjusted_img in adjusted_images:
                        new_adjusted_img = add_random_occlusion(adjusted_img.copy(), num_occlusions_list )
                        new_adjusted_images.append(new_adjusted_img)

                if operation == 17:
                    # Apply cutmix with blending alpha
                    alpha = 0.1
                    selected_images = random.sample(images, 2)
                    image1 = selected_images[0]
                    image2 = selected_images[1]
                    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
                    cutmixed_image = cutmix_images(image1.copy(), image2.copy(), alpha)
                    new_adjusted_images.append(cutmixed_image)

                if operation == 18:
                    # Apply mixup with blending alpha
                    alpha = 0.6
                    selected_images = random.sample(images, 2)
                    image1 = selected_images[0]
                    image2 = selected_images[1]
                    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
                    mixed_image = mixup_images(image1.copy(), image2.copy(), alpha)
                    new_adjusted_images.append(mixed_image)

                # Update the list of adjusted images
                adjusted_images = new_adjusted_images




            # Save processed images after all adjustments for all operations
            for i, adjusted_img in enumerate(adjusted_images):
                output_filename = f"adjusted_operations_{img_file[:-4]}_step{i}.jpg"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, adjusted_img)

# Main function that executes the entire image processing process

def main():
    # Set the input and output folder paths
    input_folder = "input_images"
    output_folder = "output_images"
    # Create the output folder (if it doesn't exist)
    os.makedirs(output_folder, exist_ok=True)
    # Initialize the operations list
    operations = []
    # Start the operation selection loop
    while True:
        print("Select one or more image processing operations:")
        print("1. Hue")
        print("2. Brightless")
        print("3. Saturation")
        print("4. Add noise")
        print("5. Contrast")
        print("6. Mosaic")
        print("7. Blur")
        print("8. Translation")
        print("9. Shearing")
        print("10. Cropping")
        print("11. Rotation")
        print("12. Scaling")
        print("13. Flipping")
        print("14. Cutout")
        print("15. Image splicing")
        print("16. Occlusion")
        print("17. Cutmix")
        print("18. Mixup")
        print("Type 'done' to finish.")
        # Input the operation chosen by the user
        operation = input("Enter the number of the operation (or 'done' to finish): ")
        # If the user enters 'done', exit the loop
        if operation == 'done':
            break
        else:
            # Convert the entered operation into an integer
            operation = int(operation)
            # Check if the operation is within valid range
            if operation in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
                # Append the operation to the operations list
                operations.append(operation)
            else:
                print("Invalid operation choice.")
    # Process the images
    process_images(input_folder, output_folder, operations)
    # Print a completion message
    print(f"Images processed and saved in {output_folder}")

# Make sure the main function is only executed when the script is run directly
if __name__ == "__main__":
    main()