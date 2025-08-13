import torch
from torch import nn
from networks.blocks import Conv2dBlock, ActFirstResBlock, DeepBLSTM, DeepGRU, DeepLSTM
from networks.utils import _len2mask, init_weights

class SharedBackbone(nn.Module):
    def __init__(self, resolution=16, max_dim=256, in_channel=1, norm='none', SN_param=False, dropout=0.0):
        super(SharedBackbone, self).__init__()
        
        # Define layer names for feature extraction
        self.layer_name_mapping = {
            '9': "feat2",
            '13': "feat3",
            '16': "feat4",
        }
        
        nf = resolution
        layers = []
        
        # Initial conv
        layers.extend([
            nn.ConstantPad2d(2, -1),
            Conv2dBlock(in_channel, nf, 5, 1, 0, norm='none', activation='none')
        ])
        
        # First two blocks
        for i in range(2):
            nf_out = min([int(nf * 2), max_dim])
            layers.extend([
                ActFirstResBlock(nf, nf, None, 'lrelu', norm, sn=SN_param, dropout=dropout / 2),
                nn.ReflectionPad2d((1, 1, 0, 0)),
                ActFirstResBlock(nf, nf_out, None, 'lrelu', norm, sn=SN_param, dropout=dropout / 2),
                nn.ReflectionPad2d(1),
                nn.MaxPool2d(kernel_size=3, stride=2)
            ])
            nf = min([nf_out, max_dim])

        # Third block
        df = nf
        df_out = min([int(df * 2), max_dim])
        layers.extend([
            ActFirstResBlock(df, df, None, 'lrelu', norm, sn=SN_param, dropout=dropout),
            ActFirstResBlock(df, df_out, None, 'lrelu', norm, sn=SN_param, dropout=dropout),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ])
        df = min([df_out, max_dim])

        # Final block
        df_out = min([int(df * 2), max_dim])
        layers.extend([
            ActFirstResBlock(df, df, None, 'lrelu', norm, sn=SN_param, dropout=dropout / 2),
            ActFirstResBlock(df, df_out, None, 'lrelu', norm, sn=SN_param, dropout=dropout / 2)
        ])
        
        self.cnn_backbone = nn.Sequential(*layers)
        self.output_dim = df_out
        
    def forward(self, x, ret_feats=False):
        if not ret_feats:
            return self.cnn_backbone(x), None
        
        feats = {}
        for i, layer in enumerate(self.cnn_backbone):
            x = layer(x)
            # Check if this layer index is in our mapping
            if str(i) in self.layer_name_mapping:
                feats[self.layer_name_mapping[str(i)]] = x
        
        return x, feats

class StyleEncoder(nn.Module):
    def __init__(self, style_dim=32, resolution=16, max_dim=256, in_channel=1, init='N02',
                 SN_param=False, norm='none', shared_backbone=None):

        super(StyleEncoder, self).__init__()
        self.reduce_len_scale = 16
        self.style_dim = style_dim

        # Use shared backbone or create own
        if shared_backbone is not None:
            self.cnn_backbone = shared_backbone
            df_out = shared_backbone.output_dim
        else:
            self.cnn_backbone = SharedBackbone(resolution, max_dim, in_channel, norm, SN_param)
            df_out = self.cnn_backbone.output_dim

        df = max_dim
        
        ######################################
        # Construct StyleEncoder head
        ######################################
        cnn_e = [nn.ReflectionPad2d((1, 1, 0, 0)),
                 Conv2dBlock(df_out, df, 3, 2, 0,
                             norm=norm,
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_wid = nn.Sequential(*cnn_e)
        self.linear_style = nn.Sequential(
            nn.Linear(df, df),
            nn.LeakyReLU()
        )
        self.mu = nn.Linear(df, style_dim)
        self.logvar = nn.Linear(df, style_dim)

        if init != 'none':
            init_weights(self, init)

        torch.nn.init.constant_(self.logvar.weight.data, 0.)
        torch.nn.init.constant_(self.logvar.bias.data, -10.)

    def forward(self, img, img_len, cnn_backbone=None, ret_feats=False, vae_mode=False):
        # Use provided backbone or own backbone
        if cnn_backbone is not None:
            feat, all_feats = cnn_backbone(img, ret_feats)
        else:
            feat, all_feats = self.cnn_backbone(img, ret_feats)
        
        img_len = img_len // self.reduce_len_scale
        out_e = self.cnn_wid(feat).squeeze(-2)
        img_len_mask = _len2mask(img_len, out_e.size(-1)).unsqueeze(1).float().detach()
        assert img_len.min() > 0, img_len.cpu().numpy()
        style = (out_e * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        style = self.linear_style(style)
        mu = self.mu(style)
        
        if vae_mode:
            logvar = self.logvar(style)
            encode_z = self.sample(mu, logvar)
            if ret_feats:
                return encode_z, mu, logvar, all_feats
            return encode_z, mu, logvar
        else:
            if ret_feats:
                return mu, all_feats
            return mu

    @staticmethod
    def sample(mu, logvar):
        std = torch.exp(0.5 * logvar)
        rand_z_score = torch.randn_like(std)
        return mu + rand_z_score * std


class WriterIdentifier(nn.Module):
    def __init__(self, n_writer=284, resolution=16, max_dim=256, in_channel=1, init='N02',
                 SN_param=False, dropout=0.0, norm='bn', shared_backbone=None):

        super(WriterIdentifier, self).__init__()
        self.reduce_len_scale = 16

        # Use shared backbone or create own
        if shared_backbone is not None:
            self.cnn_backbone = shared_backbone
            df_out = shared_backbone.output_dim
        else:
            self.cnn_backbone = SharedBackbone(resolution, max_dim, in_channel, norm, SN_param, dropout)
            df_out = self.cnn_backbone.output_dim

        df = max_dim

        ######################################
        # Construct WriterIdentifier head
        ######################################
        cnn_w = [nn.ReflectionPad2d((1, 1, 0, 0)),
                 Conv2dBlock(df_out, df, 3, 2, 0,
                             norm=norm,
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_wid = nn.Sequential(*cnn_w)
        self.linear_wid = nn.Sequential(
            nn.Linear(df, df),
            nn.LeakyReLU(),
            nn.Linear(df, n_writer),
        )

        if init != 'none':
            init_weights(self, init)

    def forward(self, img, img_len, cnn_backbone=None, ret_feats=False):
        # Use provided backbone or own backbone
        if cnn_backbone is not None:
            feat, all_feats = cnn_backbone(img, ret_feats)
        else:
            feat, all_feats = self.cnn_backbone(img, ret_feats)
        
        img_len = img_len // self.reduce_len_scale
        out_w = self.cnn_wid(feat).squeeze(-2)
        img_len_mask = _len2mask(img_len, out_w.size(-1)).unsqueeze(1).float().detach()
        wid_feat = (out_w * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        wid_logits = self.linear_wid(wid_feat)
        
        if ret_feats:
            return wid_logits, all_feats
        return wid_logits


class Recognizer(nn.Module):
    # resolution: 32  max_dim: 512  in_channel: 1  norm: 'none'  init: 'N02'  dropout: 0.  n_class: 72  rnn_depth: 0
    def __init__(self, n_class, resolution=16, max_dim=256, in_channel=1, norm='none',
                 init='none', rnn_depth=1, dropout=0.0, bidirectional=True):
        super(Recognizer, self).__init__()
        self.len_scale = 8
        self.use_rnn = rnn_depth > 0
        self.bidirectional = bidirectional

        ######################################
        # Construct Backbone
        ######################################
        nf = resolution
        cnn_f = [nn.ConstantPad2d(2, -1),
                 Conv2dBlock(in_channel, nf, 5, 1, 0,
                             norm='none',
                             activation='none')]
        for i in range(2):
            nf_out = min([int(nf * 2), max_dim])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d(1)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            nf = min([nf_out, max_dim])

        df = nf
        for i in range(2):
            df_out = min([int(df * 2), max_dim])
            cnn_f += [ActFirstResBlock(df, df, None, 'relu', norm, 'zero', dropout=dropout)]
            cnn_f += [ActFirstResBlock(df, df_out, None, 'relu', norm, 'zero', dropout=dropout)]
            if i < 1:
                cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            else:
                cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            df = min([df_out, max_dim])

        ######################################
        # Construct Classifier
        ######################################
        cnn_c = [nn.ReLU(),
                 Conv2dBlock(df, df, 3, 1, 0,
                             norm=norm,
                             activation='relu')]

        self.cnn_backbone = nn.Sequential(*cnn_f)
        self.cnn_ctc = nn.Sequential(*cnn_c)
        if self.use_rnn:
            if bidirectional:
                self.rnn_ctc = DeepBLSTM(df, df, rnn_depth, bidirectional=True)
            else:
                self.rnn_ctc = DeepLSTM(df, df, rnn_depth)
        self.ctc_cls = nn.Linear(df, n_class)

        if init != 'none':
            init_weights(self, init)

    def forward(self, x, x_len=None):
        cnn_feat = self.cnn_backbone(x)
        cnn_feat2 = self.cnn_ctc(cnn_feat)
        ctc_feat = cnn_feat2.squeeze(-2).transpose(1, 2)
        if self.use_rnn:
            if self.bidirectional:
                ctc_len = x_len // (self.len_scale  + 1e-8)
            else:
                ctc_len = None
            ctc_feat = self.rnn_ctc(ctc_feat, ctc_len)
        logits = self.ctc_cls(ctc_feat)
        if self.training:
            logits = logits.transpose(0, 1).log_softmax(2)
            logits.requires_grad_(True)
        return logits