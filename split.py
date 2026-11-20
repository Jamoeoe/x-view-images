import math
from PIL import Image

def split_image_into_n(image_name, n):
    img = Image.open(f"X_View/datasets/data/train/images/{image_name}.png")
    width, height = img.size

    # Calculate the closest square-like grid (rows x cols) for N pieces
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    tile_width = width // cols
    tile_height = height // rows

    lbl = None
    with open(f"X_View/datasets/data/train/labels/{image_name}.txt") as f:
        lbl = f.readlines()
            

    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= n:
                break
            left = c * tile_width
            upper = r * tile_height

            right = width if c == cols - 1 else (c + 1) * tile_width
            lower = height if r == rows - 1 else (r + 1) * tile_height

            tile = img.crop((left, upper, right, lower))
            tile.save(f"X_View/datasets/data/train/split_images/{image_name}_{count}.png")

            with open(f"X_View/datasets/data/train/split_labels/{image_name}_{count}.txt") as f:
                for label in lbl:
                    row = label.split(" ")
                    center_horizontal = width // row[1]
                    center_vertical = height // row[2]

                    lbl_left = center_horizontal - row[3] // 2
                    lbl_right = center_horizontal + row[3] // 2

                    lbl_upper = center_vertical + row[4] // 2
                    lbl_lower = center_vertical - row[4] // 2

                    #new_row = gen_row(left, upper, right, lower, lbl_left, lbl_upper, lbl_right, lbl_lower, row[0])

            count += 1

split_image_into_n('hurricane-florence_00000006_post_disaster', 4)