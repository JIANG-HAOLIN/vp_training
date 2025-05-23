import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights  # Import the weights enum

class TriggerPredictor(nn.Module):
    def __init__(self, 
                 num_embeddings=100, 
                 feature_dim=512, 
                 transformer_layers=6, 
                 nhead=8, 
                 max_seq_len=10, 
                 pretrained_resnet=False):
        """
        Args:
          num_embeddings: Number of possible trajectory indices (vocabulary size).
          feature_dim: Dimensionality of image features (and transformer model dimension).
          transformer_layers: Number of layers in the transformer encoder.
          nhead: Number of attention heads.
          max_seq_len: Maximum length of the input image sequence.
          pretrained_resnet: Whether to initialize ResNet18 with pretrained weights.
        """
        super(TriggerPredictor, self).__init__()
        # Choose the appropriate weights.
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_resnet else None
        # Initialize ResNet18 with the new parameter.
        resnet = models.resnet18(weights=weights)
        # Remove the final FC layer to obtain convolutional features.
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Output shape: [B, 512, 1, 1]
        self.feature_dim = feature_dim
        # Optionally add a projection if feature_dim is not 512.
        if feature_dim != 512:
            self.resnet_fc = nn.Linear(512, feature_dim)
        else:
            self.resnet_fc = None
        
        # Embedding for trajectory indices.
        self.current_area_embedding = nn.Embedding(num_embeddings, feature_dim)
        self.target_area_embedding = nn.Embedding(num_embeddings, feature_dim)
        self.vel_emb = nn.Linear(3, feature_dim)
        
        # Positional embeddings for the tokens.
        # +1 to account for the class token.
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len + 1, feature_dim))
        
        # Transformer encoder (ViT-style) to process the sequence of tokens.
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Final prediction head that maps the class token to a 3D velocity.
        self.cls_head = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())
        
    def forward(self, rgb, cur_area, target_area, vel):
        """
        Args:
          rgb: Tensor of shape [B, seq_len, 3, 224, 224].
          pose: Tensor of shape [B, seq_len, 3] (not used by this network).
          traj_idx: Tensor of shape [B] containing trajectory indices.
          
        Returns:
          velocity: Tensor of shape [B, 1] representing the predicted velocity.
        """
        B, seq_len, C, H, W = rgb.shape
        # Process images: flatten batch and sequence dimensions.
        rgb = rgb.view(B * seq_len, C, H, W)  # [B*seq_len, 3, 224, 224]
        features = self.resnet(rgb)           # [B*seq_len, 512, 1, 1]
        features = features.view(B * seq_len, 512)  # [B*seq_len, 512]
        if self.resnet_fc is not None:
            features = self.resnet_fc(features)     # [B*seq_len, feature_dim]
        # Reshape to get sequence features: [B, seq_len, feature_dim]
        img_features = features.view(B, seq_len, self.feature_dim)
        
        # Get trajectory embedding for each sample: [B, feature_dim]
        cur_area_embed = self.current_area_embedding(cur_area)
        target_area_embed = self.target_area_embedding(target_area)
        vel_embed = self.vel_emb(vel)

        # Concatenate the trajectory embedding (as class token) with image features.
        tokens = img_features + cur_area_embed + target_area_embed + vel_embed # [B, seq_len, feature_dim]
        
        # Add positional embeddings.
        # Note: We slice the positional embeddings to match the actual token sequence length.
        tokens = tokens + self.pos_embedding[:, 1:tokens.shape[1]+1, :]
        cls_token = self.pos_embedding[:, 0:1, :]
        tokens = torch.cat([cls_token.expand(B, -1, -1), tokens], dim=1)

        encoded_tokens = self.transformer_encoder(tokens)  # [B, seq_len+1, feature_dim]
        
        # Use the class token (first token) for prediction.
        class_token = encoded_tokens[:, 0, :]  # [B, feature_dim]
        pred_score = self.cls_head(class_token)        # [B, 1]
        return pred_score

# Example usage:
if __name__ == "__main__":
    # Create dummy inputs.
    B, seq_len = 4, 10
    rgb = torch.randn(B, seq_len, 3, 224, 224)
    pose = torch.randn(B, seq_len, 3)  # Not used in this network.
    cur_area = torch.randint(0, 100, [B, seq_len])
    target_area = torch.randint(0, 100, [B, seq_len])
    vel = torch.randn(B, seq_len, 3)
    # Initialize the network.
    model = TriggerPredictor(num_embeddings=100, 
                                        feature_dim=512, 
                                        transformer_layers=6, 
                                        nhead=8, 
                                        max_seq_len=seq_len,
                                        pretrained_resnet=False)
    # Forward pass.
    pred = model(rgb, cur_area, target_area, vel)
    print("Predicted score shape:", pred.shape)  # Expected shape: [B, 1]
