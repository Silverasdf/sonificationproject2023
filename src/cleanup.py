# Cleanup, Ryan Peruski, 06-21-2023
# Contains functions for dispersing, removing, combining, flipping, and splitting files. One outlier is the gradcam function
import os, sys, shutil, random
import pandas as pd
import cv2 as cv
import numpy as np
import torch
import random
import torch.nn.functional as F

def delete_files_and_copy(folder_path, destination_path, percentage, subfolder_names=['0', '1']):
    # Create destination folder if it doesn't exist
    os.makedirs(destination_path, exist_ok=True)

    for subfolder_name in subfolder_names:
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            destination_subfolder_path = os.path.join(destination_path, subfolder_name)
            # Copy subfolder to destination
            shutil.copytree(subfolder_path, destination_subfolder_path)

            files = os.listdir(destination_subfolder_path)
            num_files = len(files)
            num_files_to_delete = int(num_files * percentage / 100)
            files_to_delete = random.sample(files, num_files_to_delete)

            for file_name in files_to_delete:
                file_path = os.path.join(destination_subfolder_path, file_name)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

def iou(box1, box2):
    #Take the areas of the boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    #Find the intersection
    intersection = [max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])]
    #If there is no intersection, return 0
    if intersection[0] >= intersection[2] or intersection[1] >= intersection[3]:
        return 0
    #Otherwise, return the intersection over union
    intersection_area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
    return intersection_area / (area1 + area2 - intersection_area)

# Takes class files and disperses them into training, validation, and testing folders
def disperse_files(source_dir, target_dir):
    # Create target directories
    training_dir = os.path.join(target_dir, "Training")
    validation_dir = os.path.join(target_dir, "Validation")
    testing_dir = os.path.join(target_dir, "Testing")
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(testing_dir, exist_ok=True)

    # Iterate over class folders
    for class_folder in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_folder)
        if os.path.isdir(class_path):
            class_training_dir = os.path.join(training_dir, class_folder)
            class_validation_dir = os.path.join(validation_dir, class_folder)
            class_testing_dir = os.path.join(testing_dir, class_folder)
            os.makedirs(class_training_dir, exist_ok=True)
            os.makedirs(class_validation_dir, exist_ok=True)
            os.makedirs(class_testing_dir, exist_ok=True)

            # Get the files in the class folder
            files = os.listdir(class_path)
            print(files)
            num_files = len(files)
            random.shuffle(files)

            # Calculate the number of files for each split
            num_training = int(0.7 * num_files)
            num_validation = int(0.15 * num_files)
            num_testing = num_files - num_training - num_validation

            # Move files to the respective splits while preserving folder structure
            for i, file in enumerate(files):
                src_path = os.path.join(class_path, file)
                if i < num_training:
                    dst_path = os.path.join(class_training_dir, file)
                elif i < num_training + num_validation:
                    dst_path = os.path.join(class_validation_dir, file)
                else:
                    dst_path = os.path.join(class_testing_dir, file)
                shutil.copy(src_path, dst_path)
    print("Successfully dispersed files to: {}".format(target_dir))

# Think "rm -rf directory"
def remove_directory(directory):
    try:
        shutil.rmtree(directory)
        print(f"Successfully removed directory: {directory}")
    except OSError as e:
        print(f"Error occurred while removing directory: {directory}")
        print(e)

# Split_back takes a directory of images and a csv file containing the bounding box coordinates and splits the images using those bounding boxes into the respective folders
def split_back(data_dir, save_dir, sheet = 'MasterSheet_2023_updated.csv', res = [91, 91], bb_dir='Data/BoundingBoxesNew2023'):
    df = pd.read_csv(sheet)
    for filepath in os.listdir(data_dir):
        filename = os.path.basename(filepath)
        try:
            #Parse Filename
            new_filename = filename

            matching_row = df[df["Filename"] == new_filename]
            if matching_row.empty:
                raise Exception(f'No matching row for {new_filename}')

        except Exception as e:
            #Any weird files need to be ignored
            print(f'{e}: Did not successfully parse')
            continue

        #Now that we know the file exists, let's crop the image
        coord_list = []
        for i in np.arange(1,7):
            x1 = matching_row[f"RearSeatX{i}"].values[0]
            y1 = matching_row[f"RearSeatY{i}"].values[0]
            # If x1 or y1 as NaN, then skip
            if np.isnan(x1) or np.isnan(y1) or x1 == -1 or y1 == -1:
                if i < 2:
                    print(f'WARNING: Less than 2 seats in {new_filename}')
                break
            x1 -= res[0]/2 #Because x1 needs to be the top left corner, not the center!
            y1 -= res[1]/2
            x2 = x1 + res[0]
            y2 = y1 + res[1]
            #print(f'x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}')
            coord_list.append([x1, y1, x2, y2])
            img = cv.imread(f'{data_dir}/{filename}')
            if img is None:
                print(f'Could not read {filename}')
                break
            crop_img = img[int(y1):int(y2), int(x1):int(x2)]
            gt = int(matching_row[f"Occ{i}"].values[0])
            cv.imwrite(f'{save_dir}/{gt}/{new_filename}_backseat_passenger_{i}.jpg', crop_img)

        # Let's test the image by drawing a bounding box and saving them
        for num, coords in enumerate(coord_list):
            cv.rectangle(img, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 2)
            # Label their class
            cv.putText(img, str(matching_row[f"Occ{num+1}"].values[0]), (int(coords[0]), int(coords[1])-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv.imwrite(f'{bb_dir}/{new_filename}_backseat_passengers.jpg', img)

# Splits an image into two and saves them as separate files
def split_image_lengthwise(image_path, save_directory, left_suffix="passenger", right_suffix="driver"):
    # Load the image
    image = cv.imread(image_path)
    
    # Get the dimensions of the image
    height, width, _ = image.shape
    
    # Split the image into two equal parts
    split_point = width // 2
    passenger_half = image[:, :split_point, :]
    driver_half = image[:, split_point:, :]
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    
    # Save the passenger and driver halves as separate files
    passengerstr = os.path.join(save_directory, os.path.basename(image_path[:-4] + '_' + left_suffix + '.jpg'))
    driverstr = os.path.join(save_directory, os.path.basename(image_path[:-4] + '_' + right_suffix + '.jpg'))
    success_passenger = cv.imwrite(passengerstr, passenger_half)
    success_driver = cv.imwrite(driverstr, driver_half)
    if not success_passenger:
        print("Failed to write {} image: {}".format(left_suffix, passengerstr))
        sys.exit(1)
    if not success_driver:
        print("Failed to write {} image: {}".format(right_suffix, driverstr))
        sys.exit(1)

# Splits front images into passenger and driver halves
def split_front(save_directory, picture_directory, class_names=["0", "1"], extra=False):
    folder_names = class_names
        
    # Iterate over all files in the directory and perform the split
    for filename in os.listdir(picture_directory):
        if filename.endswith('.jpg'):
            file_path = os.path.join(picture_directory, filename)
            split_image_lengthwise(file_path, save_directory)

    # Extra thing: Moves them all into class folders randomly. For test data, see this script in test data directory, as it is done differently

    if extra:
        # Create folders
        for folder_name in folder_names:
            folder_path = os.path.join(save_directory, folder_name)
            os.makedirs(folder_path, exist_ok=True)

        # Get all files in the directory
        files = os.listdir(save_directory)

        # Iterate over each file
        for file in files: 
            # Randomly assign the file to a folder
            folder = random.choice(folder_names)
            # Get the full path of the file
            file_path = os.path.join(save_directory, file)
            # Get the destination folder path
            destination_folder = os.path.join(save_directory, folder)
            # Move the file to the destination folder
            shutil.move(file_path, destination_folder)

#Combines images from data_dir and puts into new_data_dir - looks for suffix in filename
def combine_pictures(data_dir, new_data_dir, bbdir='Data/CombinedBoundingBoxes', left_label='passenger', right_label='driver', suffix='backseat_passengers.jpg'):
    for filepath in os.listdir(new_data_dir):
        filename = os.path.basename(filepath)
        filename = 'CabinImage_Vehicle-' + filename.split('_')[1] + '_' + filename.split('_')[2] + suffix
        for filepath2 in os.listdir(data_dir):
            filename2 = os.path.basename(filepath2)
            if filename == filename2:
                img = cv.imread(f'{data_dir}/{filename}')
                img2 = cv.imread(f'{new_data_dir}/{filepath}')
                #Combine the two images
                img = np.concatenate((img, img2), axis=1)
                #Label new_data_dir and data_dir
                img = cv.putText(img, left_label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                img = cv.putText(img, right_label, (img.shape[1] - 180, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                #Save the image
                cv.imwrite(f'{bbdir}/{filename[:-4]}.jpg', img)
                break

# Flips images in a directory if they end with flip_if_ending
def flip_pictures(img_dir, flip_if_ending='passenger.jpg'):
    for root, dirs, files in os.walk(img_dir):
        # Process files
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_path = file_path.replace('\\', '/')
            # Do something with the file_path
            img = cv.imread(file_path)
            if file_path.endswith(flip_if_ending):
                img = cv.flip(img, 1)
            cv.imwrite(file_path, img)

# Tests a model on an image and returns the image
def generate_gradcam(model, image):
    # Set model to evaluation mode
    model.eval()

    # Move model to the appropriate device (GPU or CPU)
    device = next(model.parameters()).device

    # Convert image to torch tensor and move it to the same device as the model
    image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
    image = image.to(device)

    # Forward pass
    features = model(image)
    output = F.softmax(features, dim=1)
    _, predicted = torch.max(output, 1)
    predicted_class = predicted.item()

    # Get the last layer's activation
    last_layer_activation = features[0]

    # Compute gradients
    model.zero_grad()
    output[0, predicted_class].backward()

    # Get the gradients of the last layer
    gradients = model.fc.weight.grad

    # Compute the importance weights using global average pooling
    weights = torch.mean(gradients.view(gradients.size(0), gradients.size(1), -1), dim=2)

    # Multiply the weights with the last layer activation
    weighted_activation = torch.mul(last_layer_activation, weights.unsqueeze(2).unsqueeze(3))
    heatmap = torch.mean(weighted_activation, dim=1).squeeze().detach().cpu().numpy()

    # Apply ReLU and normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Resize heatmap to match the input image size
    heatmap = cv.resize(heatmap, (image.shape[3], image.shape[2]))

    # Apply colormap to the heatmap
    heatmap = cv.applyColorMap(np.uint8(255 * heatmap), cv.COLORMAP_JET)

    # Resize the original image to match the heatmap size
    resized_image = cv.resize(image.squeeze().permute(1, 2, 0).cpu().numpy(), (heatmap.shape[1], heatmap.shape[0]))

    # Convert the heatmap and resized image to the same data type
    heatmap = heatmap.astype(np.float32)
    resized_image = resized_image.astype(np.float32)

    # Overlay heatmap on the resized image
    overlaid_img = cv.addWeighted(resized_image, 0.7, heatmap, 0.3, 0, dtype=cv.CV_32F)

    return overlaid_img