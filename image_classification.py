import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
from report import create_pdf

def random_image_classification(folder_path, model, class_names, transform):
    # Get a list of image filenames in the folder
    image_filenames = os.listdir(folder_path)

    # Randomly choose an image
    random_image_filename = random.choice(image_filenames)
    random_image_path = os.path.join(folder_path, random_image_filename)

    # Open and transform the randomly chosen image
    random_image = Image.open(random_image_path).convert('L')
    random_image_tensor = transform(random_image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(random_image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_class_name = class_names[predicted_class_index]
        predicted_probability = probabilities[0, predicted_class_index].item()

    plt.imshow(random_image, cmap='gray')
    plt.title('Predicted: {}\nProbability: {:.3f}'.format(predicted_class_name, predicted_probability))
    plt.axis('off')
    plt.show()


def random_image_classification_report(folder_path, model, class_names, transform, llm, llm_model_config, report_config):
    image_filenames = os.listdir(folder_path)
    random_image_filename = random.choice(image_filenames)
    random_image_path = os.path.join(folder_path, random_image_filename)
    random_image = Image.open(random_image_path).convert('L')
    random_image_tensor = transform(random_image).unsqueeze(0)

    with torch.no_grad():
        output = model(random_image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_class_name = class_names[predicted_class_index]
        predicted_probability = probabilities[0, predicted_class_index].item()

    response = llm(llm_model_config["pre_prompt"] +
                   ' {}'.format(predicted_class_name) +
                   llm_model_config["post_prompt"])


    create_pdf(report_config["title"],
               random_image_path,
               'Chest X-ray prediction: {} with a probability: {:.3f}'.format(predicted_class_name, predicted_probability),
               response,
               'sf-labs_report.pdf')

