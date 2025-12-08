import os
import random
import shutil


# def process_images_in_directory(dir_path):
#     """
#     In the specified directory, this function renames image files
#     where the filename's index satisfies the condition (index % 4 == 1)
#     from .png to .jpg, and deletes all other image files.

#     Args:
#         dir_path (str): The path to the directory containing the images.
#     """
#     if not os.path.isdir(dir_path):
#         print(f"Error: Directory not found at '{dir_path}'")
#         return

#     # Get a list of all files in the directory
#     try:
#         filenames = os.listdir(dir_path)
#     except OSError as e:
#         print(f"Error accessing directory: {e}")
#         return

#     for filename in filenames:
#         # Construct the full path of the file
#         full_path = os.path.join(dir_path, filename)

#         # Ensure it's a file and ends with .png
#         if os.path.isfile(full_path) and filename.lower().endswith('.png'):
#             # Get the filename without the extension
#             base_name = os.path.splitext(filename)[0]

#             try:
#                 # Convert the filename to an integer index
#                 index = int(base_name)

#                 # Check if the index satisfies the condition index % 4 == 1
#                 if index % 4 == 1:
#                     # If it satisfies, rename the file extension to .jpg
#                     new_filename = f"{base_name}.jpg"
#                     new_full_path = os.path.join(dir_path, new_filename)
#                     os.rename(full_path, new_full_path)
#                     print(f"Renamed '{filename}' to '{new_filename}'")
#                 else:
#                     # If it does not satisfy, delete the file
#                     os.remove(full_path)
#                     print(f"Deleted '{filename}'")

#             except ValueError:
#                 # Handle cases where the filename is not a valid integer
#                 print(f"Skipped '{filename}': filename is not a valid integer.")
#             except OSError as e:
#                 print(f"Error processing file '{filename}': {e}")


# if __name__ == "__main__":
#     # The target directory path
#     # Using a raw string (r"...") to handle backslashes in Windows paths
#     target_directorys = [r"datasets\DefogDataset\test\ground_truth", r"datasets\DefogDataset\train\ground_truth"]

#     for target_directory in target_directorys:
#         print(f"Processing files in: {target_directory}")
#         process_images_in_directory(target_directory)
#         print("Processing complete.")



# def move_random_images(src_dir, dst_dir, num_to_move):
#     """
#     Randomly selects num_to_move images from src_dir and moves them to dst_dir.
#     If a file with the same name exists in dst_dir, appends a suffix to avoid overwriting.

#     Args:
#         src_dir (str): Source directory containing images.
#         dst_dir (str): Destination directory to move images to.
#         num_to_move (int): Number of images to move.
#     """
#     if not os.path.isdir(src_dir):
#         print(f"Source directory not found: {src_dir}")
#         return
#     if not os.path.isdir(dst_dir):
#         print(f"Destination directory not found: {dst_dir}")
#         return

#     image_files = [f for f in os.listdir(src_dir)
#                    if os.path.isfile(os.path.join(src_dir, f)) and f.lower().endswith(('.png', '.jpg'))]

#     if len(image_files) < num_to_move:
#         print(f"Not enough images to move: found {len(image_files)}, need {num_to_move}")
#         return

#     selected_files = random.sample(image_files, num_to_move)

#     for filename in selected_files:
#         src_path = os.path.join(src_dir, filename)
#         dst_path = os.path.join(dst_dir, filename)

#         # If file exists, add suffix
#         if os.path.exists(dst_path):
#             base, ext = os.path.splitext(filename)
#             suffix = 1
#             new_filename = f"{base}_{suffix}{ext}"
#             new_dst_path = os.path.join(dst_dir, new_filename)
#             while os.path.exists(new_dst_path):
#                 suffix += 1
#                 new_filename = f"{base}_{suffix}{ext}"
#                 new_dst_path = os.path.join(dst_dir, new_filename)
#             dst_path = new_dst_path

#         shutil.move(src_path, dst_path)
#         print(f"Moved '{filename}' to train directory as '{os.path.basename(dst_path)}'.")

# if __name__ == "__main__":
#     src_directory = r"datasets\DefogDataset\test\ground_truth"
#     dst_directory = r"datasets\DefogDataset\train\ground_truth"
#     move_random_images(src_directory, dst_directory, 150)



def rename_and_shuffle_images(dir_path, class_name, mode):
    if mode in dir_path:
        if not os.path.isdir(dir_path):
            print(f"Directory not found: {dir_path}")
            return

        image_files = [f for f in os.listdir(dir_path)
                    if os.path.isfile(os.path.join(dir_path, f)) and f.lower().endswith('.jpg')]

        random.shuffle(image_files)

        for idx, filename in enumerate(image_files, start=1):
            src_path = os.path.join(dir_path, filename)
            new_filename = f"{class_name}_{mode}_{idx}.jpg"
            dst_path = os.path.join(dir_path, new_filename)
            os.rename(src_path, dst_path)
            print(f"Renamed '{filename}' to '{new_filename}'")

if __name__ == "__main__":
    class_names = ['rain', 'snow', 'fog']
    for class_name in class_names:
        target_dirs = [rf"datasets\De{class_name}Dataset\test\ground_truth", rf"datasets\De{class_name}Dataset\train\ground_truth"]
        for target_dir in target_dirs:
            rename_and_shuffle_images(target_dir, class_name, 'test')
            rename_and_shuffle_images(target_dir, class_name, 'train')