import numpy as np
from PIL import Image
import pandas as pd
import math
import os
from pathlib import Path

width = 768
height = 768
counter = 0
isKeepEmpty = True

Path("./airbus-ship-detection/masks_v2").mkdir(parents=True, exist_ok=True)
def convert_to_tuples(input_array):
    if len(input_array) % 2 != 0:
        raise ValueError("Input array length must be even.")

    result_array = [(input_array[i], input_array[i + 1]) for i in range(0, len(input_array), 2)]
    return result_array

counter_empty = 0
def createMask(file_name,white_pixels_raw):
    global counter_empty
    image = np.zeros((height, width), dtype=np.uint8)
    if(len(white_pixels_raw) < 2):
        if(counter_empty >= 10 or isKeepEmpty):
            # Convert the NumPy array to a PIL Image
            pil_image = Image.fromarray(image)
            # Save the image to a file
            pil_image.save('./airbus-ship-detection/masks_v2/' + file_name)
            global counter
            counter += 1
            counter_empty += 1
            return
        else:
            print('else', file_name)
            try:
                print('else if')
                os.remove('./airbus-ship-detection/train_v2/' + file_name)
                print('removed ' + file_name)
            except OSError:
                pass
            counter_empty = 0
            return

    white_pixels = convert_to_tuples(white_pixels_raw)


    for val, amount in white_pixels:
        y = round(val / width)
        x = val % width
        positionX = 0
        for indx in range(amount):
            if x+positionX < width:
                image[x+positionX-1, y-1] = 255
            else:
                y += 1
                if y > height:
                    y = 767
                x = 0
                positionX = 1
                image[x, y-1] = 255

            positionX += 1
    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(image)
    pil_image.save('./airbus-ship-detection/masks_v2/'+file_name)
    print(counter, 'Mask created ' + file_name)
    counter += 1


dataset = pd.read_csv('./airbus-ship-detection/train_ship_segmentations_v2.csv') #.iloc[:, :].values
dataset['EncodedPixels'] = dataset['EncodedPixels'].fillna('0')
dataset['EncodedPixels'] = dataset['EncodedPixels'].str.split(' ')

dataset['EncodedPixels'] = dataset['EncodedPixels'].apply(lambda x: list(map(lambda val: int(val), x)))
dataset = dataset.iloc[:, :].values


for line in dataset:
    createMask(line[0], line[1])

# for testing
# createMask('134c8ead9.jpg', [519789, 2, 520557, 4, 521324, 7, 522092, 9, 522859, 12, 523627, 14, 524394, 17, 525162, 19, 525929, 23, 526697, 25, 527465, 27, 528232, 30, 529000, 32, 529767, 35, 530535, 37, 531302, 41, 532070, 43, 532837, 46, 533605, 48, 534373, 50, 535140, 53, 535908, 55, 536675, 58, 537443, 61, 538210, 64, 538978, 66, 539745, 69, 540513, 71, 541280, 74, 542048, 76, 542816, 78, 543583, 82, 544351, 84, 545118, 87, 545886, 89, 546653, 92, 547421, 94, 548188, 97, 548956, 100, 549726, 100, 550496, 100, 551266, 100, 552036, 100, 552806, 100, 553576, 100, 554346, 100, 555117, 100, 555887, 100, 556657, 100, 557427, 100, 558197, 100, 558967, 100, 559738, 99, 560508, 99, 561278, 100, 562048, 100, 562818, 100, 563588, 100, 564358, 100, 565129, 99, 565899, 100, 566669, 100, 567439, 100, 568209, 100, 568979, 100, 569750, 99, 570520, 99, 571290, 100, 572060, 100, 572830, 99, 573600, 97, 574370, 95, 575141, 92, 575911, 90, 576681, 88, 577451, 86, 578221, 84, 578991, 82, 579761, 80, 580532, 77, 581302, 75, 582072, 73, 582842, 71, 583612, 69, 584382, 67, 585153, 64, 585923, 62, 586693, 60, 587463, 58, 588233, 56, 589003, 54, 589773, 52])
# createMask(dataset[2][0], dataset[2][1])
# createMask('1.jpeg', [765, 100])