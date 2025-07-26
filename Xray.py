"""
Medical Imaging Feature Extractor Module

This module provides a feature extraction component for medical imaging captioning systems.
It processes batches of chest X-ray images (grayscale or RGB) using a pretrained ResNet50
backbone (without classification head) and outputs compact 2048-dimensional embeddings
suitable for feeding into an LSTM-based decoder.

Key Features:
- Supports both CPU and GPU operation
- Handles dynamic batching
- Proper image preprocessing for ResNet50
- Optional dimension projection
- Extensible architecture for other CNN backbones
"""

import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalImageFeatureExtractor:
    """
    A feature extraction component for medical imaging that uses a pretrained ResNet50 backbone.

    Args:
        device (str, optional): Device to run computations on ('cuda' or 'cpu'). 
                               Auto-detects GPU if available by default.
        projection_dim (int, optional): If specified, adds a linear projection layer to 
                                       reduce embedding dimension. Defaults to None.
        freeze_backbone (bool): Whether to freeze ResNet50 weights during extraction.
                               Defaults to True.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        projection_dim: Optional[int] = None,
        freeze_backbone: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.projection_dim = projection_dim
        self.freeze_backbone = freeze_backbone

        # Initialize model
        self.model = self._initialize_model()
        self.preprocess = self._get_preprocess_transform()

        logger.info(
            f"Feature extractor initialized on device: {self.device} "
            f"(projection: {projection_dim}, frozen: {freeze_backbone})"
        )

    def _initialize_model(self) -> nn.Module:
        """Initialize ResNet50 model without classification head."""
        # Load pretrained ResNet50 with updated weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove the classification head
        model = nn.Sequential(*list(model.children())[:-1])
        
        # Freeze backbone if required
        if self.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        # Add projection layer if specified
        if self.projection_dim is not None:
            projection = nn.Linear(2048, self.projection_dim)
            model = nn.Sequential(model, nn.Flatten(), projection)
        
        model = model.to(self.device)
        model.eval()
        
        return model

    def _get_preprocess_transform(self) -> transforms.Compose:
        """
        Get the preprocessing transform for ResNet50.
        
        Handles both grayscale (converted to 3-channel) and RGB images.
        Normalization uses ImageNet stats as the model is pretrained on it.
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    @staticmethod
    def _convert_to_pil(images: Union[list, np.ndarray, torch.Tensor]) -> list:
        """Convert input images to list of PIL Images."""
        if isinstance(images, (np.ndarray, torch.Tensor)):
            if len(images.shape) == 3:  # Single image
                images = [images]
            return [Image.fromarray(img) if not isinstance(img, Image.Image) else img 
                    for img in images]
        elif isinstance(images, list):
            return [Image.fromarray(img) if not isinstance(img, Image.Image) else img 
                    for img in images]
        else:
            raise ValueError("Input must be numpy array, torch tensor, or list of images")

    def preprocess_batch(self, images: Union[list, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess a batch of images for feature extraction.
        
        Args:
            images: Input images as either:
                   - List of PIL Images or numpy arrays
                   - numpy array of shape (batch, height, width[, channels])
                   - torch tensor of shape (batch, height, width[, channels])
        
        Returns:
            torch.Tensor: Preprocessed batch of shape (batch, 3, 224, 224)
        """
        pil_images = self._convert_to_pil(images)
        
        # Apply preprocessing to each image and stack into batch
        processed_images = torch.stack([self.preprocess(img) for img in pil_images])
        
        return processed_images.to(self.device)

    def extract_features(
        self, 
        images: Union[list, np.ndarray, torch.Tensor],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract features from a batch of medical images.
        
        Args:
            images: Input images in various formats (see preprocess_batch)
            batch_size: Batch size for processing. Adjust based on available memory.
        
        Returns:
            np.ndarray: Array of features with shape (num_images, embedding_dim)
                        where embedding_dim is 2048 or projection_dim if specified.
        """
        # Preprocess all images first
        processed_images = self.preprocess_batch(images)
        num_images = len(processed_images)
        features = []

        # Process in batches
        with torch.no_grad():
            for i in range(0, num_images, batch_size):
                batch = processed_images[i:i + batch_size]
                batch_features = self.model(batch)
                
                # Handle different output shapes based on projection
                if self.projection_dim is None:
                    batch_features = batch_features.view(batch_features.size(0), -1)
                
                features.append(batch_features.cpu().numpy())
        
        # Concatenate all batch results
        return np.concatenate(features, axis=0)

    def get_embedding_dim(self) -> int:
        """Get the output dimension of the feature embeddings."""
        return self.projection_dim if self.projection_dim is not None else 2048

    def to(self, device: str) -> None:
        """Move the feature extractor to the specified device."""
        self.device = device
        self.model = self.model.to(device)
        logger.info(f"Feature extractor moved to device: {device}")


# Example usage
if __name__ == "__main__":
    # Example initialization
    feature_extractor = MedicalImageFeatureExtractor(
        projection_dim=512,
        freeze_backbone=True
    )
    
    # Example with dummy data (in practice, load real X-ray images)
    dummy_images = [np.random.rand(256, 256) * 255 for _ in range(10)]  # 10 grayscale images
    dummy_images_rgb = [np.random.rand(256, 256, 3) * 255 for _ in range(10)]  # 10 RGB images
    
    # Extract features
    features_gray = feature_extractor.extract_features(dummy_images)
    features_rgb = feature_extractor.extract_features(dummy_images_rgb)
    
    print(f"Gray features shape: {features_gray.shape}")
    print(f"RGB features shape: {features_rgb.shape}")
