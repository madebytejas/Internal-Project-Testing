"""
Medical Imaging Feature Extractor for Radiology Diagnostics

A production-ready, modular feature extraction system designed for medical imaging
applications, specifically chest X-ray analysis. Supports ResNet50 backbone with
extensibility for other CNN architectures.

Author: AI Solutions Architect
Version: 1.0.2
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
    """
    
    def __init__(self, images: Union[List[str], List[np.ndarray], List[torch.Tensor]], 
                 transform: Optional[transforms.Compose] = None):
        self.images = images
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        image = self.images[idx]
        try:
            tensor_image = self._process_image(image)
            if self.transform:
                tensor_image = self.transform(tensor_image)
            return tensor_image
        except Exception as e:
            logger.error(f"Error processing image at index {idx}: {str(e)}")
            raise IOError(f"Failed to process image at index {idx}: {str(e)}")

    def _process_image(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Process different image types into tensor"""
        if isinstance(image, str):
            return self._load_image_file(image)
        elif isinstance(image, np.ndarray):
            return self._process_numpy_array(image)
        elif isinstance(image, torch.Tensor):
            return self._process_torch_tensor(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _load_image_file(self, image_path: str) -> torch.Tensor:
        """Load image from file path"""
        pil_image = Image.open(image_path).convert('RGB')
        return transforms.ToTensor()(pil_image)

    def _process_numpy_array(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor"""
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

    def _process_torch_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """Process torch tensor into correct format"""
        if image.dim() == 2:
            tensor_image = image.unsqueeze(0).repeat(3, 1, 1)
        elif image.dim() == 3:
            if image.size(0) == 1:
                tensor_image = image.repeat(3, 1, 1)
            elif image.size(2) == 3:
                tensor_image = image.permute(2, 0, 1)
            else:
                tensor_image = image
        else:
            tensor_image = image
            
        if tensor_image.max() > 1.0:
            tensor_image = tensor_image / 255.0
        return tensor_image


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
        self.model_name = model_name
        self.pretrained = pretrained
        self.feature_dim = feature_dim
        self.input_size = input_size
        self.normalize_features = normalize_features
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size


class MedicalImageFeatureExtractor(nn.Module):
    """Production-ready medical image feature extractor."""
    
    def __init__(self, config: FeatureExtractorConfig):
        super(MedicalImageFeatureExtractor, self).__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone = self._create_backbone()
        self.feature_projector = self._create_feature_projector()
        self.preprocess = self._create_preprocessing_pipeline()
        self.to(self.device)
        
    def _create_backbone(self) -> nn.Module:
        if self.config.model_name.lower() == 'resnet50':
            model = models.resnet50(pretrained=self.config.pretrained)
            return nn.Sequential(*list(model.children())[:-1])
        elif self.config.model_name.lower() == 'densenet121':
            model = models.densenet121(pretrained=self.config.pretrained)
            model.classifier = nn.Identity()
            return model
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")

    def _create_feature_projector(self) -> nn.Sequential:
        backbone_out = self._get_backbone_output_dim()
        return nn.Sequential(
            nn.Linear(backbone_out, self.config.feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.feature_dim * 2, self.config.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_rate)
        )

    def _get_backbone_output_dim(self) -> int:
        dummy = torch.randn(1, 3, *self.config.input_size)
        with torch.no_grad():
            output = self.backbone(dummy)
            return output.view(output.size(0), -1).size(1)

    def _create_preprocessing_pipeline(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(self.config.input_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = self.feature_projector(features)
        if self.config.normalize_features:
            features = F.normalize(features, p=2, dim=1)
        return features

    def extract_features_batch(self, images: List[torch.Tensor], batch_size: int = None) -> torch.Tensor:
        batch_size = batch_size or self.config.batch_size
        dataset = MedicalImageDataset(images, transform=self.preprocess)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.eval()
        with torch.no_grad():
            features = torch.cat([self(batch.to(self.device)).cpu() for batch in dataloader])
        return features

    def extract_features_single(self, image: torch.Tensor) -> torch.Tensor:
        dataset = MedicalImageDataset([image], transform=self.preprocess)
        self.eval()
        with torch.no_grad():
            return self(dataset[0].unsqueeze(0).to(self.device)).cpu()

    def save_model(self, filepath: Union[str, Path]) -> None:
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__
        }, filepath)

    @classmethod
    def load_model(cls, filepath: Union[str, Path], device: str = None) -> 'MedicalImageFeatureExtractor':
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)
        config = FeatureExtractorConfig(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        if device:
            model.device = torch.device(device)
            model.to(model.device)
        return model

    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'model_name': self.config.model_name,
            'feature_dim': self.config.feature_dim,
            'input_size': self.config.input_size,
            'device': str(self.device),
            'total_params': total_params
        }


def create_feature_extractor(model_name: str = 'resnet50',
                           feature_dim: int = 2048,
                           pretrained: bool = True,
                           **kwargs) -> MedicalImageFeatureExtractor:
    config = FeatureExtractorConfig(
        model_name=model_name,
        feature_dim=feature_dim,
        pretrained=pretrained,
        **kwargs
    )
    return MedicalImageFeatureExtractor(config)


if __name__ == "__main__":
    print("=== Medical Image Feature Extractor ===")
    
    # Initialize extractor
    extractor = create_feature_extractor()
    print("\nModel created successfully!")
    
    # Show model info
    info = extractor.get_model_info()
    print("\nModel Information:")
    for k, v in info.items():
        print(f"{k:>15}: {v}")
    
    # Create test data (proper format: 3 channels, 224x224)
    test_images = [torch.rand(3, 224, 224) for _ in range(4)]
    print(f"\nCreated {len(test_images)} test images (3, 224, 224)")
    
    # Extract features
    print("\nExtracting features...")
    features = extractor.extract_features_batch(test_images)
    print(f"Features shape: {features.shape}")
    print(f"First feature vector (first 5 values): {features[0][:5].tolist()}")
    
    # Save model
    extractor.save_model("medical_feature_extractor.pth")
    print("\nModel saved to 'medical_feature_extractor.pth'")
    
    print("\n=== Test Completed Successfully ===")
