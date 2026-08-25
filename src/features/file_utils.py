import os


def ensure_path_exists(folder, filename):

    full_path = os.path.join(
        folder,
        filename
    )

    os.makedirs(
        folder,
        exist_ok=True
    )

    return full_path