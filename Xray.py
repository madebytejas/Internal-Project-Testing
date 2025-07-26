"""
Medical Imaging Feature Extractor for Radiology Diagnostics

A production-ready, modular feature extraction system designed for medical imaging
applications, specifically chest X-ray analysis. Supports ResNet50 backbone with
extensibility for other CNN architectures.

Author: AI Solutions Architect
Version: 1.0.1
License: MIT
"""

import logging
import warnings
from typing import Union, List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


class MedicalImageDataset(Dataset):
    """
    Custom dataset for medical images with preprocessing capabilities.
    
    Handles both single images and batch processing with proper medical imaging
    preprocessing standards.
    """
    
    def __init__(self, images: Union[List[str], List[np.ndarray], List[torch.Tensor]], 
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize the dataset.
        
        Args:
            images: List of image paths, numpy arrays, or torch tensors
            transform: Optional preprocessing transforms
        """
        self.images = images
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get preprocessed image tensor.
        
        Args:
            idx: Index of the image
            
        Returns:
            Preprocessed image tensor
            
        Raises:
            ValueError: If image format is not supported
            IOError: If image file cannot be loaded
        """
        image = self.images[idx]
        
        try:
            if isinstance(image, str):
                # Load from file path
                pil_image = Image.open(image).convert('RGB')
                tensor_image = transforms.ToTensor()(pil_image)
            elif isinstance(image, np.ndarray):
                # Convert numpy array to tensor
                if image.ndim == 2:  # Grayscale
                    image = np.stack([image] * 3, axis=-1)  # Convert to RGB
                elif image.ndim == 3 and image.shape[-1] == 1:  # Single channel
                    image = np.repeat(image, 3, axis=-1)  # Convert to RGB
                
                tensor_image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            elif isinstance(image, torch.Tensor):
                # Handle torch tensor - ensure channels-first format
                if image.dim() == 2:  # Grayscale (H, W)
                    tensor_image = image.unsqueeze(0).repeat(3, 1, 1)  # -> (3, H, W)
                elif image.dim() == 3:
                    if image.size(0) == 1:  # Single channel (1, H, W)
                        tensor_image = image.repeat(3, 1, 1)  # -> (3, H, W)
                    elif image.size(2) == 3:  # Channels-last (H, W, C)
                        tensor_image = image.permute(2, 0, 1)  # -> (C, H, W)
                    else:
                        tensor_image = image
                else:
                    tensor_image = image
                    
                if tensor_image.max() > 1.0:
                    tensor_image = tensor_image / 255.0
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Apply transforms if provided
            if self.transform:
                tensor_image = self.transform(tensor_image)
                
            return tensor_image
            
        except Exception as e:
            logger.error(f"Error processing image at index {idx}: {str(e)}")
            raise IOError(f"Failed to process image at index {idx}: {str(e)}")


class FeatureExtractorConfig:
    """Configuration class for the feature extractor."""
    
    def __init__(self, 
                 model_name: str = 'resnet50',
                 pretrained: bool = True,
                 feature_dim: int = 2048,
                 input_size: Tuple[int, int] = (224, 224),
                 normalize_features: bool = True,
                 dropout_rate: float = 0.1,
                 batch_size: int = 32):
        """
        Initialize configuration.
        
        Args:
            model_name: Name of the backbone model
            pretrained: Whether to use pretrained weights
            feature_dim: Output feature dimension
            input_size: Input image size (height, width)
            normalize_features: Whether to L2 normalize features
            dropout_rate: Dropout rate for regularization
            batch_size: Default batch size for processing
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.feature_dim = feature_dim
        self.input_size = input_size
        self.normalize_features = normalize_features
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size


class MedicalImageFeatureExtractor(nn.Module):
    """
    Production-ready medical image feature extractor.
    
    Designed for chest X-ray analysis with ResNet50 backbone and extensibility
    for other CNN architectures. Outputs compact embeddings suitable for
    LSTM-based decoders in medical captioning systems.
    """
    
    def __init__(self, config: FeatureExtractorConfig):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration object containing model parameters
            
        Raises:
            ValueError: If unsupported model architecture is specified
        """
        super(MedicalImageFeatureExtractor, self).__init__()
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize backbone model
        self.backbone = self._create_backbone()
        
        # Feature projection layer
        backbone_out_features = self._get_backbone_output_dim()
        self.feature_projector = nn.Sequential(
            nn.Linear(backbone_out_features, config.feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.feature_dim * 2, config.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate)
        )
        
        # Medical imaging specific preprocessing
        self.preprocess = self._create_preprocessing_pipeline()
        
        # Move model to appropriate device
        self.to(self.device)
        
        logger.info(f"Initialized {config.model_name} feature extractor on {self.device}")
        logger.info(f"Output feature dimension: {config.feature_dim}")
        
    def _create_backbone(self) -> nn.Module:
        """
        Create the backbone CNN model.
        
        Returns:
            Backbone model without classification head
            
        Raises:
            ValueError: If unsupported model architecture is specified
        """
        if self.config.model_name.lower() == 'resnet50':
            model = models.resnet50(pretrained=self.config.pretrained)
            # Remove classification head
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.config.model_name.lower() == 'resnet101':
            model = models.resnet101(pretrained=self.config.pretrained)
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.config.model_name.lower() == 'resnet152':
            model = models.resnet152(pretrained=self.config.pretrained)
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.config.model_name.lower() == 'densenet121':
            model = models.densenet121(pretrained=self.config.pretrained)
            model.classifier = nn.Identity()
        elif self.config.model_name.lower() == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=self.config.pretrained)
            model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model architecture: {self.config.model_name}")
            
        return model
    
    def _get_backbone_output_dim(self) -> int:
        """
        Get the output dimension of the backbone model.
        
        Returns:
            Output feature dimension of the backbone
        """
        if 'resnet' in self.config.model_name.lower():
            return 2048
        elif 'densenet121' in self.config.model_name.lower():
            return 1024
        elif 'efficientnet_b0' in self.config.model_name.lower():
            return 1280
        else:
            # Dynamic dimension detection
            dummy_input = torch.randn(1, 3, *self.config.input_size)
            with torch.no_grad():
                output = self.backbone(dummy_input)
                return output.view(output.size(0), -1).size(1)
    
    def _create_preprocessing_pipeline(self) -> transforms.Compose:
        """
        Create medical imaging specific preprocessing pipeline.
        
        Returns:
            Preprocessing transform pipeline
        """
        return transforms.Compose([
            transforms.Resize(self.config.input_size),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet standards
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.
        
        Args:
            x: Input image tensor [batch_size, channels, height, width]
            
        Returns:
            Feature embeddings [batch_size, feature_dim]
            
        Raises:
            RuntimeError: If forward pass fails
        """
        try:
            # Extract features using backbone
            features = self.backbone(x)
            
            # Flatten features
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            
            # Project to desired dimension
            features = self.feature_projector(features)
            
            # Optional L2 normalization
            if self.config.normalize_features:
                features = F.normalize(features, p=2, dim=1)
            
            return features
            
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise RuntimeError(f"Feature extraction failed: {str(e)}")
    
    def extract_features_batch(self, 
                             images: Union[List[str], List[np.ndarray], List[torch.Tensor]],
                             batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Extract features from a batch of images.
        
        Args:
            images: List of images (file paths, numpy arrays, or tensors)
            batch_size: Batch size for processing (optional)
            
        Returns:
            Feature embeddings [num_images, feature_dim]
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If feature extraction fails
        """
        if not images:
            raise ValueError("Empty image list provided")
        
        batch_size = batch_size or self.config.batch_size
        
        try:
            # Create dataset and dataloader
            dataset = MedicalImageDataset(images, transform=self.preprocess)
            dataloader = DataLoader(dataset, batch_size=batch_size, 
                                  shuffle=False, num_workers=2, pin_memory=True)
            
            all_features = []
            
            self.eval()
            with torch.no_grad():
                for batch in dataloader:
                    batch = batch.to(self.device, non_blocking=True)
                    features = self(batch)
                    all_features.append(features.cpu())
            
            return torch.cat(all_features, dim=0)
            
        except Exception as e:
            logger.error(f"Batch feature extraction failed: {str(e)}")
            raise RuntimeError(f"Batch processing failed: {str(e)}")
    
    def extract_features_single(self, 
                              image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from a single image.
        
        Args:
            image: Single image (file path, numpy array, or tensor)
            
        Returns:
            Feature embedding [1, feature_dim]
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If feature extraction fails
        """
        try:
            # Process single image
            dataset = MedicalImageDataset([image], transform=self.preprocess)
            image_tensor = dataset[0].unsqueeze(0).to(self.device)
            
            self.eval()
            with torch.no_grad():
                features = self(image_tensor)
            
            return features.cpu()
            
        except Exception as e:
            logger.error(f"Single image feature extraction failed: {str(e)}")
            raise RuntimeError(f"Single image processing failed: {str(e)}")
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
            
        Raises:
            IOError: If model saving fails
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            save_dict = {
                'model_state_dict': self.state_dict(),
                'config': self.config.__dict__,
                'model_info': {
                    'model_name': self.config.model_name,
                    'feature_dim': self.config.feature_dim,
                    'device': str(self.device)
                }
            }
            
            torch.save(save_dict, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")
            raise IOError(f"Failed to save model: {str(e)}")
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path], 
                   device: Optional[str] = None) -> 'MedicalImageFeatureExtractor':
        """
        Load a pre-trained model.
        
        Args:
            filepath: Path to the saved model
            device: Target device ('cpu' or 'cuda')
            
        Returns:
            Loaded feature extractor model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Reconstruct config
            config_dict = checkpoint['config']
            config = FeatureExtractorConfig(**config_dict)
            
            # Create model instance
            model = cls(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Move to specified device
            if device:
                model.device = torch.device(device)
                model.to(model.device)
            
            logger.info(f"Model loaded from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.config.model_name,
            'feature_dimension': self.config.feature_dim,
            'input_size': self.config.input_size,
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'normalize_features': self.config.normalize_features,
            'dropout_rate': self.config.dropout_rate
        }


# Factory function for easy model creation
def create_feature_extractor(model_name: str = 'resnet50',
                           feature_dim: int = 2048,
                           pretrained: bool = True,
                           **kwargs) -> MedicalImageFeatureExtractor:
    """
    Factory function to create a feature extractor with common configurations.
    
    Args:
        model_name: Backbone model architecture
        feature_dim: Output feature dimension
        pretrained: Whether to use pretrained weights
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured feature extractor instance
        
    Example:
        >>> extractor = create_feature_extractor('resnet50', feature_dim=2048)
        >>> features = extractor.extract_features_single('chest_xray.jpg')
    """
    config = FeatureExtractorConfig(
        model_name=model_name,
        feature_dim=feature_dim,
        pretrained=pretrained,
        **kwargs
    )
    
    return MedicalImageFeatureExtractor(config)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    try:
        # Create feature extractor
        extractor = create_feature_extractor(
            model_name='resnet50',
            feature_dim=2048,
            pretrained=True,
            normalize_features=True
        )
        
        # Print model information
        info = extractor.get_model_info()
        print("Model Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Example with properly shaped dummy data (channels-first format)
        dummy_images = [torch.randn(3, 224, 224) for _ in range(5)]  # (C, H, W) format
        
        # Extract features from batch
        batch_features = extractor.extract_features_batch(dummy_images, batch_size=2)
        print(f"\nBatch features shape: {batch_features.shape}")
        
        # Extract features from single image
        single_features = extractor.extract_features_single(dummy_images[0])
        print(f"Single image features shape: {single_features.shape}")
        
        # Save model
        extractor.save_model("medical_feature_extractor.pth")
        print("\nModel saved successfully!")
        
    except Exception as e:
        logger.error(f"Example execution failed: {str(e)}")
        print(f"Error: {str(e)}")
