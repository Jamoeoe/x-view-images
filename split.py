import math
import os
from PIL import Image

def gen_row(tile_left, tile_upper, tile_right, tile_lower, lbl_left, lbl_upper, lbl_right, lbl_lower, category):
    # Find the intersection between the bounding box and the current tile
    inter_left = max(tile_left, lbl_left)
    inter_upper = max(tile_upper, lbl_upper)
    inter_right = min(tile_right, lbl_right)
    inter_lower = min(tile_lower, lbl_lower)

    # Check if the bounding box actually overlaps with this tile
    if inter_right <= inter_left or inter_lower <= inter_upper:
        return None

    # Filter out tiny edge slivers (keep only if >= 10% of original box area remains)
    orig_area = (lbl_right - lbl_left) * (lbl_lower - lbl_upper)
    inter_area = (inter_right - inter_left) * (inter_lower - inter_upper)
    if orig_area > 0 and (inter_area / orig_area) < 0.10:
        return None

    # Calculate new dimensions relative to the current tile
    current_tile_w = tile_right - tile_left
    current_tile_h = tile_lower - tile_upper

    new_w = (inter_right - inter_left) / current_tile_w
    new_h = (inter_lower - inter_upper) / current_tile_h
    new_x_center = ((inter_left + inter_right) / 2.0 - tile_left) / current_tile_w
    new_y_center = ((inter_upper + inter_lower) / 2.0 - tile_upper) / current_tile_h

    return f"{category} {new_x_center:.6f} {new_y_center:.6f} {new_w:.6f} {new_h:.6f}\n"


def split_image_into_n_power4(image_name, n):
    # Validation: Ensure n is a power of 4 (1, 4, 16, 64, ...)
    if n < 1:
        raise ValueError("Number of splits must be at least 1.")
    
    # Mathematical check: log base 4 of n must be an integer
    log4 = math.log(n, 4)
    if not math.isclose(log4, round(log4)):
        # Find the next higher power of 4 to suggest to the user
        next_n = 4 ** math.ceil(log4)
        raise ValueError(f"Invalid split count: {n}. Must be a power of 4 (e.g., 1, 4, 16, 64, 256). Did you mean {next_n}?")

    img_path = f"X_View/datasets/data/train/images/{image_name}.png"
    lbl_path = f"X_View/datasets/data/train/labels/{image_name}.txt"
    
    out_img_dir = f"X_View/datasets/data/train/split_{n}_images"
    out_lbl_dir = f"X_View/datasets/data/train/split_{n}_labels"
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    img = Image.open(img_path)
    width, height = img.size

    # For N = 4^k, the grid is always an equal (2^k) x (2^k) layout
    grid_size = int(math.sqrt(n))  # e.g., N=1 -> 1x1, N=4 -> 2x2, N=16 -> 4x4, N=64 -> 8x8
    
    tile_width = width // grid_size
    tile_height = height // grid_size

    with open(lbl_path, "r") as f:
        lbl = f.readlines()

    count = 0
    for r in range(grid_size):
        for c in range(grid_size):
            left = c * tile_width
            upper = r * tile_height
            # Ensure the rightmost and bottommost edges line up perfectly with the original boundaries
            right = width if c == grid_size - 1 else (c + 1) * tile_width
            lower = height if r == grid_size - 1 else (r + 1) * tile_height

            # Crop and save the image tile
            tile = img.crop((left, upper, right, lower))
            tile.save(f"{out_img_dir}/{image_name}_{count}.png")

            # Process and write split labels
            with open(f"{out_lbl_dir}/{image_name}_{count}.txt", "w") as f_out:
                for label in lbl:
                    if not label.strip():
                        continue
                    row = label.strip().split()
                    cat = row[0]
                    
                    # Convert normalized YOLO format coordinates back to absolute pixels
                    center_horizontal = float(row[1]) * width
                    center_vertical = float(row[2]) * height
                    box_w = float(row[3]) * width
                    box_h = float(row[4]) * height

                    # Calculate absolute box boundaries (Y=0 is top in PIL)
                    lbl_left = center_horizontal - (box_w / 2.0)
                    lbl_right = center_horizontal + (box_w / 2.0)
                    lbl_upper = center_vertical - (box_h / 2.0)
                    lbl_lower = center_vertical + (box_h / 2.0)

                    new_row = gen_row(left, upper, right, lower, lbl_left, lbl_upper, lbl_right, lbl_lower, cat)
                    if new_row:
                        f_out.write(new_row)

            count += 1

# Example usage for a 16-way split (4x4 grid)
#split_image_into_n_power4('hurricane-florence_00000006_post_disaster', 16)

images_to_split = f"X_View/datasets/data/train/images/"

with os.scandir(images_to_split) as files:
    for file in files:
        img_name = file.name[:-4]
        split_image_into_n_power4(img_name, 16)

            