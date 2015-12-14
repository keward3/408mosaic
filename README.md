# 408mosaic

This project aims to create parallel code to create a visual representation of a given image as a grid of smaller images.

SDL and SDL_image libraries are needed for compilation.

On running the program, the original.png file will be used as the original image, and all images in the Images directory will be used to compose the final output image.  Samples are included in this repository that can be used to test the program.

References: Our resizing kernel and cuda error checking function are heavily based on code from here: http://mkaczanowski.com/bilinear-interpolation-with-nvidia-cuda-c/

Team members: Kyler Ward & Jacob Palecki
