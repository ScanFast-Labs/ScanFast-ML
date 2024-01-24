from config_loader import load_config
from data_utils import get_medmnist_info, get_transforms
from model_utils import load_image_classifier, load_llm_model
from image_classification import random_image_classification_report, random_image_classification
import sys

def main():
    config = load_config("config.yaml")
    medmnist_info, _ = get_medmnist_info(config["medmnist_config"])
    n_classes = len(medmnist_info["label"])
    test_images = config["file_paths"]["test_images_path"]
    class_names = config["medmnist_config"]["class_names"]

    model = load_image_classifier(config["image_classifier_model_config"], n_classes)
    transform = get_transforms()

    # Check the command line argument to determine the mode of execution
    if len(sys.argv) > 1:
        if sys.argv[1] == "image-only":
            # Run only the image classification part
            random_image_classification(test_images, model, class_names, transform)
        elif sys.argv[1] == "full-scan":
            llm = load_llm_model(config["llm_model_config"])
            llm_model_config = config["llm_model_config"]
            report_config = config["report_config"]
            random_image_classification_report(test_images, model, class_names, transform, llm, llm_model_config, report_config)
    else:
        # If no arguments are provided, provide an error message
        print("Error: Please specify 'image-only' for image classification or 'full-scan' for a report based on image classification and medical llm.")
        sys.exit(1)

if __name__ == "__main__":
    main()
