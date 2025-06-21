import os

class Utils:

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    IMAGE_DIR = os.path.join(PROJECT_ROOT, 'images')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'compressed_images')

    DEFAULT_F = 8
    DEFAULT_D = 10

    WINDOW_SIZE = "1200x800"