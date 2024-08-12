from PIL import Image, ImageDraw
import numpy as np

# Load the image
image = Image.open('/home/jiexiao/gaussian-splatting/analysis/model01/m01v000.png')

# Image dimensions
width, height = image.size

# Calculate the coordinates of the pixel based on its index
pixel_index = 282777
x = pixel_index % width
y = pixel_index // width

print(f"Coordinates of the pixel: ({x}, {y})")

# Convert the image to a NumPy array
image_array = np.array(image)

# Extract the pixel value
pixel_value = image_array[y, x]

print(f"Pixel value at ({x}, {y}): {pixel_value}")

# Get the 8 neighboring pixels
neighbors = []
for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        if dx == 0 and dy == 0:
            continue
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            neighbors.append(image_array[ny, nx])

print(f"Neighboring pixels: {neighbors}")

# Create a new image with the enlarged pixel and its neighbors
zoom_factor = 100  # Adjust this to change the enlargement size
enlarged_pixels = np.zeros((3 * zoom_factor, 3 * zoom_factor, 3), dtype=np.uint8)

for i, (dx, dy) in enumerate([(0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]):
    nx, ny = x + dx, y + dy
    if 0 <= nx < width and 0 <= ny < height:
        enlarged_pixels[(dy + 1) * zoom_factor:(dy + 2) * zoom_factor,
                        (dx + 1) * zoom_factor:(dx + 2) * zoom_factor] = image_array[ny, nx]

# Convert the enlarged pixel array back to an image
enlarged_image = Image.fromarray(enlarged_pixels)

# Draw a circle around the pixel in the original image
draw = ImageDraw.Draw(image)
radius = 5  # Radius of the circle to highlight the pixel
draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline='red', width=2)

# Display the original image with the circled pixel
image.show()

# Save the original image with the circled pixel
image.save('circled_pixel_image.jpg')

# Save or display the enlarged image
enlarged_image.save('enlarged_pixel_and_neighbors.png')
enlarged_image.show()
