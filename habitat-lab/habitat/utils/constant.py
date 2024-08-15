coco_categories = [0, 3, 2, 4, 5, 1]
category_to_id = [
        "chair",
        "bed",
        "plant",
        "toilet",
        "tv_monitor",
        "sofa"
]

category_to_id = [
        "chair",
        "bed",
        "plant",
        "toilet",
        "tv_monitor",
        "sofa"
]
hm3d_category = [
        "chair",
        "sofa",
        "plant",
        "bed",
        "toilet",
        "tv_monitor",
        "bathtub",
        "shower",
        "fireplace",
        "appliances",
        "towel",
        "sink",
        "chest_of_drawers",
        "table",
        "stairs"
]

def parse_tensor_value(tensor_value):
    # Ensure tensor is moved to CPU and converted to a NumPy array
    array_value = tensor_value.cpu().numpy()

    # Check if the array is 2D or 1D
    if array_value.ndim == 2:
        # For 2D tensor
        value_only_string = '\n'.join(
            [' '.join([f'{item:.1f}' for item in row]) for row in array_value])
    elif array_value.ndim == 1:
        # For 1D tensor
        value_only_string = ' '.join([f'{item:.1f}' for item in array_value])
    else:
        # For scalar values
        value_only_string = f'{array_value:.1f}'

    return value_only_string
