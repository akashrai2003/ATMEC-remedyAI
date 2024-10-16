import GANsImage  # Assuming GAN.py is the file where you have defined your functions

def test_generate_images_gan_chest():
    num_images = 20  # Number of images to generate for testing
    try:
        zip_file_path = GANsImage.generate_images_gan_chest(num_images)
        print(f"Chest GAN: Images successfully generated and saved in {zip_file_path}")
    except Exception as e:
        print(f"Chest GAN: An error occurred during generation: {e}")

def test_generate_images_gan_knee():
    num_images = 20  # Number of images to generate for testing
    try:
        zip_file_path = GANsImage.generate_images_gan_knee(num_images)
        print(f"Knee GAN: Images successfully generated and saved in {zip_file_path}")
    except Exception as e:
        print(f"Knee GAN: An error occurred during generation: {e}")

if __name__ == "__main__":
    print("Testing Chest GAN Image Generation")
    test_generate_images_gan_chest()

    print("\nTesting Knee GAN Image Generation")
    test_generate_images_gan_knee()
