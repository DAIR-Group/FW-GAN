import torch
import torch.nn as nn
import torch.nn.functional as F

class FDL_loss(nn.Module):
    def __init__(self, backbone, patch_size=2, stride=1, num_proj=512, phase_weight=1.0, upscale_factor=4, chunk_size=64):
        """
        backbone: SharedBackbone instance to extract features
        patch_size, stride, num_proj: SWD slice parameters
        phase_weight: weight for phase branch
        upscale_factor: Factor to upscale input images (reduced from 4 to 2)
        chunk_size: Number of projections to process at once
        """
        super(FDL_loss, self).__init__()
        self.backbone = backbone
        self.phase_weight = phase_weight
        self.stride = stride
        self.patch_size = patch_size
        self.num_proj = num_proj  # Reduced from 1024 to 256
        self.upscale_factor = upscale_factor  # Reduced from 4 to 2
        self.chunk_size = chunk_size
        self.chns = None
        self.initialized = False

    def initialize_projections(self, feats):
        # Convert dictionary to list if needed
        if isinstance(feats, dict):
            feat_list = list(feats.values())
        else:
            feat_list = feats
            
        self.chns = [feat.shape[1] for feat in feat_list]
        device = feat_list[0].device
        for i, chn in enumerate(self.chns):
            rand = torch.randn(self.num_proj, chn, self.patch_size, self.patch_size, device=device)
            rand = rand / rand.view(rand.size(0), -1).norm(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            self.register_buffer(f"rand_{i}", rand)
        self.initialized = True

    def forward_once_chunked(self, x, y, idx):
        """Process projections in chunks to reduce memory usage"""
        rand = self.__getattr__(f"rand_{idx}")
        
        # Process in chunks
        num_chunks = (self.num_proj + self.chunk_size - 1) // self.chunk_size
        scores = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, self.num_proj)
            rand_chunk = rand[start_idx:end_idx]
            
            # Conv operations on chunk
            projx = F.conv2d(x, rand_chunk, stride=self.stride)
            projy = F.conv2d(y, rand_chunk, stride=self.stride)
            
            # Reshape and sort
            projx = projx.reshape(projx.shape[0], projx.shape[1], -1)
            projy = projy.reshape(projy.shape[0], projy.shape[1], -1)
            
            projx, _ = torch.sort(projx, dim=-1)
            projy, _ = torch.sort(projy, dim=-1)
            
            chunk_score = torch.abs(projx - projy).mean([1, 2])
            scores.append(chunk_score)
            
            # Clear intermediate tensors to free memory
            del projx, projy, rand_chunk
            
        return torch.stack(scores, dim=1).mean(dim=1)

    def forward(self, x, y):
        """
        x, y: input images with shape (N, C, H, W)
        """
        # Reduced upscale factor to save memory
        if self.upscale_factor > 1:
            x = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
            y = F.interpolate(y, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        
        # Extract features from inputs
        _, x_feats = self.backbone(x, ret_feats=True)
        _, y_feats = self.backbone(y, ret_feats=True)
        
        # Convert dictionaries to lists if needed
        if isinstance(x_feats, dict):
            x_feat_list = list(x_feats.values())
            y_feat_list = list(y_feats.values())
        else:
            x_feat_list = x_feats
            y_feat_list = y_feats
        
        if not self.initialized:
            self.initialize_projections(x_feat_list)
        
        assert len(x_feat_list) == len(y_feat_list) == len(self.chns), "Mismatch in feature layers"
        
        total_score = 0
        for i in range(len(x_feat_list)):
            feat_x, feat_y = x_feat_list[i], y_feat_list[i]
            
            # Skip very large feature maps to prevent OOM
            if feat_x.numel() > 2e6:  # If feature map is too large
                feat_x = F.avg_pool2d(feat_x, 2)
                feat_y = F.avg_pool2d(feat_y, 2)
            
            # FFT operations
            fft_x = torch.fft.fftn(feat_x, dim=(-2, -1))
            fft_y = torch.fft.fftn(feat_y, dim=(-2, -1))

            x_mag = torch.abs(fft_x)
            x_phase = torch.angle(fft_x)
            y_mag = torch.abs(fft_y)
            y_phase = torch.angle(fft_y)

            # Use chunked processing
            s_amplitude = self.forward_once_chunked(x_mag, y_mag, i)
            s_phase = self.forward_once_chunked(x_phase, y_phase, i)
            
            layer_score = s_amplitude + s_phase * self.phase_weight
            total_score += layer_score.mean()
            
            # Clear intermediate tensors
            del fft_x, fft_y, x_mag, x_phase, y_mag, y_phase, feat_x, feat_y

        return total_score / len(x_feat_list)