
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from utils import create_model, load_state_dict

class StylePyramidInference:
    def __init__(self, checkpoint_path, cfg_path='configs/inference.yaml', device=None):
        """
        Initializes the StylePyramidInference class.

        Args:
            checkpoint_path (str): Path to the pre-trained model checkpoint.
            cfg_path (str): Path to the model configuration file.
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to GPU if available.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        self.cfg_path = cfg_path
        self.model = self._load_model()

    def _load_model(self):
        """
        Private method to load the pre-trained StylePyramid model.

        Returns:
            torch.nn.Module: The control model loaded from the checkpoint.
        """
        model = create_model(self.cfg_path).to(self.device)
        model.load_state_dict(load_state_dict(self.checkpoint_path), strict=False)
        model.eval()  # Set the model to evaluation mode
        return model.control_model

    @staticmethod
    def preprocess_image(image_path, image_size=256):
        """
        Preprocess the input image.

        Args:
            image_path (str): Path to the input image.
            image_size (int, optional): Size to which the image will be resized. Defaults to 256.

        Returns:
            torch.Tensor: Preprocessed image tensor with a batch dimension.
        """
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image

    def infer(self, image_path):
        """
        Perform inference on the input image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            torch.Tensor: Output of the model for the given image.
        """
        image = self.preprocess_image(image_path).to(self.device)
        with torch.no_grad():
            output = self.model(image)
        return output

# Example usage
if __name__ == "__main__":
    # Define paths
    image_path = "/disk512gb/TextCtr/TextCtrl/example/i_s/00000.png"  # Path to input image
    checkpoint_path = "/disk512gb/TextCtr/TextCtrl/weights/model.pth"  # Path to pre-trained model checkpoint

    # Create an instance of the inference class
    inference_model = StylePyramidInference(checkpoint_path)

    # Perform inference
    vit_output = inference_model.infer(image_path)

    # Output shape
    print("ViT Output Shape:", vit_output.shape)

    # print(vit_output.shape)


