import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPModel, CLIPVisionModel, CLIPProcessor
from torchvision.models import resnet50
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# === MODEL ===   
# class ContributionAwareModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
#         # self.text_fc = nn.Linear(768, 512)

#         self.text_encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
#         self.text_fc = nn.Linear(768, 512)

#         # self.image_encoder = resnet50(pretrained=True)
#         # self.image_encoder.fc = nn.Linear(2048, 1024)
#         # self.image_fc = nn.Linear(1024, 512) # her modality icin fc layer var, bunlarin ciktisini aliyoruz

#         self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
#         self.image_fc = nn.Linear(self.image_encoder.config.hidden_size, 512)

#         self.evidence_encoder = self.text_encoder
#         # self.evidence_encoder = AutoModel.from_pretrained("bert-base-uncased")
#         self.evidence_fc = nn.Linear(768, 512)

#         # self.classifier = nn.Linear(512, 3)
#         self.classifier = nn.Linear(512, 2)


#         # Contribution scorers 
#         self.text_scorer = ContributionScorer()
#         self.image_scorer = ContributionScorer()
#         self.evidence_scorer = ContributionScorer()

#     def forward(self, text, evidence, image):
#         text_feat = self.text_fc(self.text_encoder(**text).last_hidden_state[:, 0, :])
#         evidence_feat = self.evidence_fc(self.evidence_encoder(**evidence).last_hidden_state[:, 0, :])
        
#         # b, n, c, h, w = image.shape
#         # image = image.view(-1, c, h, w)
#         # image_feat = self.image_fc(self.image_encoder(image).view(b, n, -1).max(dim=1).values)

#         # Input shape: (B, N, C, H, W)
#         b, n, c, h, w = image.shape
#         image = image.view(-1, c, h, w)  # [B*N, C, H, W]
#         image_outputs = self.image_encoder(pixel_values=image)
#         image_feat = image_outputs.pooler_output  # [B*N, D]

#         # Reshape back to (B, N, D) and pool
#         image_feat = image_feat.view(b, n, -1).max(dim=1).values
#         image_feat = self.image_fc(image_feat)

#         # Contributions
#         c_text = self.text_scorer(text_feat)
#         c_image = self.image_scorer(image_feat)
#         c_evidence = self.evidence_scorer(evidence_feat)

#         # Stack and normalize scores
#         contribs = torch.cat([c_text, c_image, c_evidence], dim=1)
#         weights = F.softmax(contribs, dim=1).unsqueeze(-1)

#         # Stack features and weight them
#         feats = torch.stack([text_feat, image_feat, evidence_feat], dim=1)  # 3D tensor
#         fused = torch.sum(weights * feats, dim=1)  # [B, 512]

#         logits = self.classifier(fused)
#         return logits, weights.squeeze(-1)  # logits: [B, 3], contrib_scores: [B, 3]


class ContributionAwareModel(nn.Module):
    def __init__(self, fusion_type="contribution"):
        super().__init__()
        self.fusion_type = fusion_type  # "contribution" or "cross_attention"

        # === Text / Evidence Encoder ===
        self.text_encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
        self.text_fc = nn.Linear(768, 512)

        self.evidence_encoder = self.text_encoder  # share weights
        self.evidence_fc = nn.Linear(768, 512)

        # === Image Encoder (CLIP) ===
        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_fc = nn.Linear(self.image_encoder.config.hidden_size, 512)

        # === Fusion Module ===
        if self.fusion_type == "cross_attention":
            encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
            self.cross_modal_transformer = TransformerEncoder(encoder_layer, num_layers=1)

        elif self.fusion_type == "contribution":
            self.text_scorer = ContributionScorer()
            self.image_scorer = ContributionScorer()
            self.evidence_scorer = ContributionScorer()

        # === Classifier ===
        self.classifier = nn.Linear(512, 2)

    def forward(self, text, evidence, image):
        # Encode text and evidence
        text_feat = self.text_fc(self.text_encoder(**text).last_hidden_state[:, 0, :])
        evidence_feat = self.evidence_fc(self.evidence_encoder(**evidence).last_hidden_state[:, 0, :])

        # Encode image
        b, n, c, h, w = image.shape
        image = image.view(-1, c, h, w)
        image_outputs = self.image_encoder(pixel_values=image)
        image_feat = image_outputs.pooler_output
        image_feat = image_feat.view(b, n, -1).max(dim=1).values
        image_feat = self.image_fc(image_feat)

        # === Fusion: Cross-Attention ===
        if self.fusion_type == "cross_attention":
            stacked = torch.stack([text_feat, image_feat, evidence_feat], dim=1)  # [B, 3, 512]
            attended = self.cross_modal_transformer(stacked.transpose(0, 1)).transpose(0, 1)  # [B, 3, 512]
            fused = attended.mean(dim=1)  # mean-pool

            logits = self.classifier(fused)
            contrib_dummy = torch.ones(b, 3, device=fused.device) / 3  # dummy for compatibility
            return logits, contrib_dummy

        # === Fusion: Scalar Contribution Weighting ===
        elif self.fusion_type == "contribution":
            c_text = self.text_scorer(text_feat)
            c_image = self.image_scorer(image_feat)
            c_evidence = self.evidence_scorer(evidence_feat)

            contribs = torch.cat([c_text, c_image, c_evidence], dim=1)  # [B, 3]
            weights = F.softmax(contribs, dim=1).unsqueeze(-1)  # [B, 3, 1]

            feats = torch.stack([text_feat, image_feat, evidence_feat], dim=1)  # [B, 3, 512]
            fused = torch.sum(weights * feats, dim=1)  # [B, 512]

            logits = self.classifier(fused)
            return logits, weights.squeeze(-1)
    
class ContributionScorer(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # output: between 0-1
        )

    def forward(self, x):
        return self.score(x)

# For class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weighting factor per class
        self.gamma = gamma  # Focus parameter

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # Probabilities of correct predictions
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_factor = self.alpha[targets]  # Apply per-class weighting
            focal_loss = alpha_factor * focal_loss

        return focal_loss.mean()