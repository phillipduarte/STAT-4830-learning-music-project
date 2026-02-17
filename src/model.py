"""
Residual CNN for sheet music page matching.
Processes measure-level features to produce page embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ResidualBlock(nn.Module):
    """
    Residual block for 1D convolutions over measure sequences.
    
    Architecture:
        x → Conv1D → BatchNorm → ReLU → Conv1D → BatchNorm → (+) → ReLU
        └────────────────────────────────────────────────────────┘
                           (skip connection)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, downsample: Optional[nn.Module] = None):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            stride: Stride for convolution (>1 for downsampling)
            downsample: Optional layer to match dimensions for skip connection
        """
        super().__init__()
        
        # Calculate padding to maintain sequence length (when stride=1)
        padding = kernel_size // 2
        
        # Main path (two convolutions)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection (identity or projection)
        self.downsample = downsample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, in_channels, seq_len)
            
        Returns:
            Output tensor of shape (batch, out_channels, seq_len')
        """
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
            
        # Add skip connection and apply activation
        out += identity
        out = self.relu(out)
        
        return out


class ResNetPageEncoder(nn.Module):
    """
    ResNet-based encoder for sheet music pages.
    
    Takes measure-level features and produces a fixed-size page embedding.
    """
    
    def __init__(self, feature_dim: int = 25, embedding_dim: int = 128,
                 num_blocks_per_stage: Tuple[int, ...] = (2, 2, 2, 2),
                 base_channels: int = 64):
        """
        Args:
            feature_dim: Dimension of input features per measure
            embedding_dim: Dimension of output page embedding
            num_blocks_per_stage: Number of residual blocks at each stage
            base_channels: Number of channels in first stage (doubles each stage)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        # Initial convolution (projects features to base_channels)
        self.conv1 = nn.Conv1d(feature_dim, base_channels, kernel_size=7, 
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Build residual stages
        self.in_channels = base_channels
        self.stage1 = self._make_stage(base_channels, num_blocks_per_stage[0], stride=1)
        self.stage2 = self._make_stage(base_channels * 2, num_blocks_per_stage[1], stride=2)
        self.stage3 = self._make_stage(base_channels * 4, num_blocks_per_stage[2], stride=2)
        self.stage4 = self._make_stage(base_channels * 8, num_blocks_per_stage[3], stride=2)
        
        # Global pooling and final projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 8, embedding_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_stage(self, out_channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        """
        Create a stage with multiple residual blocks.
        
        Args:
            out_channels: Number of output channels for this stage
            num_blocks: Number of residual blocks in this stage
            stride: Stride for first block (for downsampling)
            
        Returns:
            Sequential container of residual blocks
        """
        downsample = None
        
        # If dimensions change, we need a projection for the skip connection
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        # First block (may downsample)
        layers.append(ResidualBlock(self.in_channels, out_channels, stride=stride,
                                   downsample=downsample))
        
        # Remaining blocks
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a page (sequence of measure features) into an embedding.
        
        Args:
            x: Input tensor of shape (batch, num_measures, feature_dim)
            
        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        # Transpose to (batch, feature_dim, num_measures) for Conv1d
        x = x.transpose(1, 2)
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Global pooling (reduce sequence dimension)
        x = self.global_pool(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)
        
        # Final projection to embedding space
        x = self.fc(x)
        
        return x


class SiamesePageMatcher(nn.Module):
    """
    Siamese network for page matching.
    
    Uses shared ResNet encoder to embed two pages, then computes similarity.
    """
    
    def __init__(self, feature_dim: int = 25, embedding_dim: int = 128,
                 num_blocks_per_stage: Tuple[int, ...] = (2, 2, 2, 2),
                 base_channels: int = 64, similarity_metric: str = 'cosine'):
        """
        Args:
            feature_dim: Dimension of input features per measure
            embedding_dim: Dimension of page embeddings
            num_blocks_per_stage: Residual blocks per stage
            base_channels: Base number of channels
            similarity_metric: How to compute similarity ('cosine', 'euclidean', or 'learned')
        """
        super().__init__()
        
        # Shared encoder for both pages
        self.encoder = ResNetPageEncoder(
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            num_blocks_per_stage=num_blocks_per_stage,
            base_channels=base_channels
        )
        
        self.similarity_metric = similarity_metric
        
        # For learned similarity, add a small MLP
        if similarity_metric == 'learned':
            self.similarity_head = nn.Sequential(
                nn.Linear(embedding_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
    
    def forward(self, page1: torch.Tensor, page2: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between two pages.
        
        Args:
            page1: Tensor of shape (batch, num_measures, feature_dim)
            page2: Tensor of shape (batch, num_measures, feature_dim)
            
        Returns:
            Similarity scores of shape (batch,) in range [0, 1]
        """
        # Encode both pages
        emb1 = self.encoder(page1)
        emb2 = self.encoder(page2)
        
        # Compute similarity
        if self.similarity_metric == 'cosine':
            # Cosine similarity: emb1 · emb2 / (||emb1|| ||emb2||)
            similarity = F.cosine_similarity(emb1, emb2, dim=1)
            # Map from [-1, 1] to [0, 1]
            similarity = (similarity + 1) / 2
            
        elif self.similarity_metric == 'euclidean':
            # Euclidean distance, converted to similarity
            distance = torch.norm(emb1 - emb2, p=2, dim=1)
            # Convert distance to similarity (exp(-d))
            similarity = torch.exp(-distance)
            
        elif self.similarity_metric == 'learned':
            # Concatenate embeddings and pass through MLP
            combined = torch.cat([emb1, emb2], dim=1)
            similarity = self.similarity_head(combined).squeeze(-1)
            
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
            
        return similarity
    
    def get_embeddings(self, pages: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for a batch of pages (useful for optimization stage).
        
        Args:
            pages: Tensor of shape (batch, num_measures, feature_dim)
            
        Returns:
            Embeddings of shape (batch, embedding_dim)
        """
        return self.encoder(pages)


def create_model(config: dict) -> SiamesePageMatcher:
    """
    Factory function to create model from config dict.
    
    Args:
        config: Dictionary with model hyperparameters
        
    Returns:
        Initialized SiamesePageMatcher model
    """
    return SiamesePageMatcher(
        feature_dim=config.get('feature_dim', 25),
        embedding_dim=config.get('embedding_dim', 128),
        num_blocks_per_stage=config.get('num_blocks_per_stage', (2, 2, 2, 2)),
        base_channels=config.get('base_channels', 64),
        similarity_metric=config.get('similarity_metric', 'cosine')
    )


if __name__ == "__main__":
    # Test the model
    print("Testing ResNet Page Matcher...")
    
    # Create dummy data
    batch_size = 4
    num_measures = 32  # 32 measures per page
    feature_dim = 25
    
    page1 = torch.randn(batch_size, num_measures, feature_dim)
    page2 = torch.randn(batch_size, num_measures, feature_dim)
    
    # Create model
    config = {
        'feature_dim': 25,
        'embedding_dim': 128,
        'num_blocks_per_stage': (2, 2, 2, 2),  # ResNet-18 style
        'base_channels': 64,
        'similarity_metric': 'cosine'
    }
    
    model = create_model(config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # Forward pass
    with torch.no_grad():
        # Test similarity computation
        similarity = model(page1, page2)
        print(f"\nSimilarity scores shape: {similarity.shape}")
        print(f"Similarity values: {similarity}")
        
        # Test embedding extraction
        embeddings = model.get_embeddings(page1)
        print(f"\nEmbedding shape: {embeddings.shape}")
        print(f"Embedding norm: {torch.norm(embeddings, dim=1)}")
    
    print("\n✓ Model test passed!")
