import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from munch import Munch
from itertools import chain
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.nn import CTCLoss, CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from fid_kid.fid_kid import calculate_kid_fid
from networks.utils import _info, set_requires_grad, get_scheduler, idx_to_words, words_to_images, rand_clip
from networks.BigGAN_networks import Generator, Discriminator, HFDiscriminator
from networks.module import Recognizer, WriterIdentifier, StyleEncoder, SharedBackbone
from lib.datasets import get_dataset, get_collect_fn, Hdf5Dataset
from lib.alphabet import strLabelConverter, get_lexicon, get_true_alphabet, Alphabets
from lib.utils import draw_image, get_logger, AverageMeterManager, option_to_string, pad
from networks.rand_dist import prepare_z_dist, prepare_y_dist
from networks.loss import FDL_loss

# wandb.login()  # Use environment variable or config file
# wandb.init(project="FW-GAN", name="FW-GAN-training", resume="allow", sync_tensorboard=True)


class BaseModel(object):
    def __init__(self, opt, log_root='./kaggle/working/'):
        self.opt = opt
        self.device = torch.device(opt.device)
        self.models = Munch()
        self.models_ema = Munch()
        self.log_root = log_root
        self.logger = None
        self.writer = None
        alphabet_key = 'rimes_word' if opt.dataset.startswith('rimes') else 'all'
        self.alphabet = Alphabets[alphabet_key]
        self.label_converter = strLabelConverter(alphabet_key)
        self.collect_fn = get_collect_fn(opt.training.sort_input)

    def print(self, info):
        if self.logger is None:
            print(info)
        else:
            self.logger.info(info)

    def create_logger(self):
        if self.logger or self.writer:
            return

        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)
        self.writer = SummaryWriter(log_dir=self.log_root)
        opt_str = option_to_string(self.opt)
        with open(os.path.join(self.log_root, 'config.txt'), 'w') as f:
            f.writelines(opt_str)
        self.logger = get_logger(self.log_root)

    def info(self, extra=None):
        self.print("RUNDIR: {}".format(self.log_root))
        opt_str = option_to_string(self.opt)
        self.print(opt_str)
        for model in self.models.values():
            self.print(_info(model, ret=True))
        if extra is not None:
            self.print(extra)
        self.print('=' * 20)

    def save(self, tag='best', epoch_done=0, **kwargs):
        ckpt = {}
        if len(self.models_ema.values()) == 0:
            for model in self.models.values():
                ckpt[type(model).__name__] = model.state_dict()
        else:
            for model in self.models_ema.values():
                ckpt[type(model).__name__] = model.state_dict()

        for key, val in kwargs.items():
            ckpt[key] = val

        ckpt['Epoch'] = epoch_done
        ckpt_save_path = os.path.join(self.log_root, self.opt.training.ckpt_dir, tag + '.pth')
        torch.save(ckpt, ckpt_save_path)

    def load(self, ckpt, map_location=None, modules=None):
        if modules is None:
            modules = []
        elif not isinstance(modules, list):
            modules = [modules]

        print('load checkpoint from ', ckpt)
        if map_location is None:
            ckpt = torch.load(ckpt)
        else:
            ckpt = torch.load(ckpt, map_location=map_location, weights_only = False)

        if len(modules) == 0:
            for model in self.models.values():
                model.load_state_dict(ckpt[type(model).__name__])
        else:
            for model in modules:
                model.load_state_dict(ckpt[type(model).__name__])

        return ckpt['Epoch']

    def set_mode(self, mode='eval'):
        for model in self.models.values():
            if mode == 'eval':
                model.eval()
            elif mode == 'train':
                model.train()
            else:
                raise NotImplementedError()

    def validate(self):
        yield NotImplementedError()

    def train(self):
        yield NotImplementedError()


class AdversarialModel(BaseModel):
    def __init__(self, opt, log_root='./kaggle/working/'):
        super(AdversarialModel, self).__init__(opt, log_root)

        device = self.device
        self.lexicon = get_lexicon(self.opt.training.lexicon,
                                   get_true_alphabet(opt.dataset),
                                   max_length=self.opt.training.max_word_len)
        self.max_valid_image_width = self.opt.char_width * self.opt.training.max_word_len
        self.noise_dim = self.opt.GenModel.style_dim - self.opt.EncModel.style_dim

        generator = Generator(**opt.GenModel).to(device)
        style_encoder = StyleEncoder(**opt.EncModel).to(device)
        writer_identifier = WriterIdentifier(**opt.WidModel).to(device)
        discriminator = Discriminator(**opt.DiscModel).to(device)
        hf_discriminator = HFDiscriminator(**opt.HFDiscModel).to(device)
        recognizer = Recognizer(**opt.OcrModel).to(device)
        shared_backbone = SharedBackbone(**opt.SharedBackbone).to(device)
        
        self.models = Munch(
            G=generator,
            D=discriminator,
            HF_D=hf_discriminator,
            R=recognizer,
            E=style_encoder,
            W=writer_identifier,
            S=shared_backbone
        )

        self.ctc_loss = CTCLoss(zero_infinity=True, reduction='mean')
        self.classify_loss = CrossEntropyLoss()
        self.fdl_loss_fn = FDL_loss(backbone=self.models.S).to(device)

    def train(self, epoch_done):
        self.info()
        
        
        def KLloss(mu, logvar):
            return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        opt = self.opt
        self.z = prepare_z_dist(opt.training.batch_size, opt.GenModel.style_dim, self.device,
                                seed=self.opt.seed)
        self.y = prepare_y_dist(opt.training.batch_size, len(self.lexicon), self.device, seed=self.opt.seed)

        self.eval_z = prepare_z_dist(opt.training.eval_batch_size, opt.GenModel.style_dim, self.device,
                                     seed=self.opt.seed)
        self.eval_y = prepare_y_dist(opt.training.eval_batch_size, len(self.lexicon), self.device,
                                     seed=self.opt.seed)

        self.train_loader = DataLoader(
            get_dataset(opt.dataset, opt.training.dset_split),
            batch_size=opt.training.batch_size,
            shuffle=True,
            collate_fn=self.collect_fn,
            num_workers=4,
            drop_last=True
        )

        self.tst_loader = DataLoader(
            get_dataset(opt.dataset, opt.valid.dset_split),
            batch_size=opt.training.eval_batch_size // 2,
            shuffle=True,
            collate_fn=self.collect_fn
        )

        self.tst_loader2 = DataLoader(
            get_dataset(opt.dataset, opt.training.dset_split),
            batch_size=opt.training.eval_batch_size // 2,
            shuffle=True,
            collate_fn=self.collect_fn,
            num_workers=4
        )

        self.optimizers = Munch(
            G=torch.optim.Adam(chain(self.models.G.parameters(), self.models.E.parameters()),
                               lr=opt.training.lr, betas=(opt.training.adam_b1, opt.training.adam_b2)),
            D=torch.optim.Adam(
                chain(self.models.D.parameters(), self.models.R.parameters(), self.models.W.parameters(), self.models.S.parameters()),
                lr=opt.training.lr, betas=(opt.training.adam_b1, opt.training.adam_b2)),
        )

        self.lr_schedulers = Munch(
            G=get_scheduler(self.optimizers.G, opt.training),
            D=get_scheduler(self.optimizers.D, opt.training)
        )

        # Updated to include 'fdl_loss' in averager_meters
        self.averager_meters = AverageMeterManager(['adv_loss', 'fake_disc_loss',
                                                    'real_disc_loss', 'hf_fake_disc_loss', 'hf_real_disc_loss', 'info_loss',
                                                    'fake_ctc_loss', 'real_ctc_loss',
                                                    'fake_wid_loss', 'real_wid_loss',
                                                    'kl_loss', 'fdl_loss', 'gp_ctc', 'gp_info', 'gp_wid'])
        device = self.device

        ctc_len_scale = 8
        best_kid = np.inf
        iter_count = 0
        for epoch in range(epoch_done, self.opt.training.epochs):
            for i, (imgs, img_lens, lbs, lb_lens, wids) in enumerate(self.train_loader):
                #############################
                # Prepare inputs & Network Forward
                #############################
                self.set_mode('train')
                real_imgs, real_img_lens, real_wids = imgs.to(device), img_lens.to(device), wids.to(device)
                real_lbs, real_lb_lens = lbs.to(device), lb_lens.to(device)

                #############################
                # Optimizing Recognizer & Writer Identifier & Discriminator
                #############################
                self.optimizers.D.zero_grad()
                set_requires_grad([self.models.G, self.models.E], False)
                set_requires_grad([self.models.R, self.models.D, self.models.HF_D, self.models.W, self.models.S], True)

                ### Compute CTC loss for real samples###
                real_ctc = self.models.R(real_imgs)
                real_ctc_lens = real_img_lens // ctc_len_scale
                real_ctc_loss = self.ctc_loss(real_ctc, real_lbs, real_ctc_lens, real_lb_lens)
                self.averager_meters.update('real_ctc_loss', real_ctc_loss.item())

                clip_imgs, clip_img_lens = rand_clip(real_imgs, real_img_lens)
                real_wid_logits = self.models.W(clip_imgs, clip_img_lens, self.models.S)
                real_wid_loss = self.classify_loss(real_wid_logits, real_wids)
                self.averager_meters.update('real_wid_loss', real_wid_loss.item())

                with torch.no_grad():
                    self.y.sample_()
                    sampled_words = idx_to_words(self.y, self.lexicon, self.opt.training.capitalize_ratio)
                    fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
                    fake_lbs, fake_lb_lens = fake_lbs.to(device).detach(), fake_lb_lens.to(device).detach()

                    self.z.sample_()
                    fake_imgs = self.models.G(self.z, fake_lbs, fake_lb_lens)

                    enc_styles, _, _ = self.models.E(real_imgs, real_img_lens,
                                                     self.models.S, vae_mode=True)
                    noises = torch.randn((real_imgs.size(0), self.opt.GenModel.style_dim
                                          - self.opt.EncModel.style_dim)).float().to(device)
                    enc_z = torch.cat([noises, enc_styles], dim=-1)
                    style_imgs = self.models.G(enc_z, fake_lbs, fake_lb_lens)

                    cat_fake_imgs = torch.cat([fake_imgs, style_imgs], dim=0)
                    cat_fake_lb_lens = fake_lb_lens.repeat(2,).detach()
                    cat_fake_img_lens = cat_fake_lb_lens * self.opt.char_width

                ### Compute discriminative loss for real & fake samples ###
                # Normal discriminator
                fake_disc = self.models.D(cat_fake_imgs.detach(), cat_fake_img_lens, cat_fake_lb_lens)
                fake_disc_loss = torch.mean(F.relu(1.0 + fake_disc))

                real_disc = self.models.D(real_imgs, real_img_lens, real_lb_lens)
                real_disc_loss = torch.mean(F.relu(1.0 - real_disc))

                # HF discriminator
                hf_fake_disc = self.models.HF_D(cat_fake_imgs.detach(), cat_fake_img_lens, cat_fake_lb_lens)
                hf_fake_disc_loss = torch.mean(F.relu(1.0 + hf_fake_disc))

                hf_real_disc = self.models.HF_D(real_imgs, real_img_lens, real_lb_lens)
                hf_real_disc_loss = torch.mean(F.relu(1.0 - hf_real_disc))

                # Combined discriminator loss
                disc_loss = (real_disc_loss + fake_disc_loss + hf_real_disc_loss + hf_fake_disc_loss)
                
                self.averager_meters.update('real_disc_loss', real_disc_loss.item())
                self.averager_meters.update('fake_disc_loss', fake_disc_loss.item())
                self.averager_meters.update('hf_real_disc_loss', hf_real_disc_loss.item())
                self.averager_meters.update('hf_fake_disc_loss', hf_fake_disc_loss.item())

                (real_ctc_loss + disc_loss + real_wid_loss).backward()
                self.optimizers.D.step()

                #############################
                # Optimizing Generator
                #############################
                if iter_count % self.opt.training.num_critic_train == 0:
                    self.optimizers.G.zero_grad()
                    set_requires_grad([self.models.D, self.models.HF_D, self.models.R, self.models.W, self.models.S], False)
                    set_requires_grad([self.models.G, self.models.E], True)

                    ##########################
                    # Prepare Fake Inputs
                    ##########################
                    self.y.sample_()
                    sampled_words = idx_to_words(self.y, self.lexicon, self.opt.training.capitalize_ratio)
                    fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
                    fake_lbs, fake_lb_lens = fake_lbs.to(device).detach(), fake_lb_lens.to(device).detach()
                    fake_img_lens = fake_lb_lens * self.opt.char_width

                    self.z.sample_()
                    fake_imgs = self.models.G(self.z, fake_lbs, fake_lb_lens)

                    enc_styles, enc_mu, enc_logvar = self.models.E(real_imgs, real_img_lens,
                                                                   self.models.S, vae_mode=True)
                    noises = torch.randn((real_imgs.size(0), self.opt.GenModel.style_dim
                                          - self.opt.EncModel.style_dim)).float().to(device)
                    enc_z = torch.cat([noises, enc_styles], dim=-1)
                    style_imgs = self.models.G(enc_z, fake_lbs, fake_lb_lens)
                    style_img_lens = fake_lb_lens * self.opt.char_width

                    ### Concatenating all generated images in a batch ###
                    cat_fake_imgs = torch.cat([fake_imgs, style_imgs], dim=0)
                    cat_fake_lbs = fake_lbs.repeat(2, 1).detach()
                    cat_fake_lb_lens = fake_lb_lens.repeat(2,).detach()
                    cat_fake_img_lens = cat_fake_lb_lens * self.opt.char_width

                    recn_imgs = self.models.G(enc_z, real_lbs, real_lb_lens)

                    ###################################################
                    # Calculating G Losses
                    ####################################################
                    ### Compute Adversarial loss ###
                    cat_fake_disc = self.models.D(cat_fake_imgs, cat_fake_img_lens, cat_fake_lb_lens)
                    adv_loss = -torch.mean(cat_fake_disc)

                    ### CTC Auxiliary loss ###
                    cat_fake_ctc = self.models.R(cat_fake_imgs)
                    cat_fake_ctc_lens = cat_fake_img_lens // ctc_len_scale
                    fake_ctc_loss = self.ctc_loss(cat_fake_ctc, cat_fake_lbs,
                                                  cat_fake_ctc_lens, cat_fake_lb_lens)

                    ### Latent Style Reconstruction ###
                    styles = self.models.E(fake_imgs, fake_img_lens, self.models.S)
                    info_loss = torch.mean(torch.abs(styles - self.z[:, -self.opt.EncModel.style_dim:].detach()))

                    ### Writer Identify Loss ###
                    recn_wid_logits = self.models.W(style_imgs, style_img_lens, self.models.S)
                    fake_wid_loss = self.classify_loss(recn_wid_logits, real_wids)

                    ### FDL Loss ###
                    fdl_loss = self.fdl_loss_fn(real_imgs, recn_imgs)

                    ### KL-Divergence Loss ###
                    kl_loss = KLloss(enc_mu, enc_logvar)

                    ### Gradient balance ###
                    grad_fake_adv = torch.autograd.grad(adv_loss, cat_fake_imgs, create_graph=True, retain_graph=True)[0]
                    grad_fake_OCR = torch.autograd.grad(fake_ctc_loss, cat_fake_ctc, create_graph=True, retain_graph=True)[0]
                    grad_fake_info = torch.autograd.grad(info_loss, fake_imgs, create_graph=True, retain_graph=True)[0]
                    grad_fake_wid = torch.autograd.grad(fake_wid_loss, recn_wid_logits, create_graph=True, retain_graph=True)[0]
                    # grad_fake_fdl = torch.autograd.grad(fdl_loss, recn_imgs, create_graph=True, retain_graph=True)[0]

                    std_grad_adv = torch.std(grad_fake_adv)
                    gp_ctc = (torch.div(std_grad_adv, torch.std(grad_fake_OCR) + 1e-8).detach() + 1).clamp_max(100)
                    gp_info = (torch.div(std_grad_adv, torch.std(grad_fake_info) + 1e-8).detach() + 1).clamp_max(50)
                    gp_wid = (torch.div(std_grad_adv, torch.std(grad_fake_wid) + 1e-8).detach() + 1).clamp_max(10)
                    # gp_fdl = (torch.div(std_grad_adv, torch.std(grad_fake_fdl) + 1e-8).detach() + 1)

                    self.averager_meters.update('gp_ctc', gp_ctc.item())
                    self.averager_meters.update('gp_info', gp_info.item())
                    self.averager_meters.update('gp_wid', gp_wid.item())
                    # self.averager_meters.update('gp_fdl', gp_fdl.item())

                    g_loss = (2 * adv_loss +
                              gp_ctc * fake_ctc_loss +
                              gp_info * info_loss +
                              gp_wid * fake_wid_loss +
                              fdl_loss +
                              self.opt.training.lambda_kl * kl_loss)
                    
                    g_loss.backward()
                    self.averager_meters.update('adv_loss', adv_loss.item())
                    self.averager_meters.update('fake_ctc_loss', fake_ctc_loss.item())
                    self.averager_meters.update('info_loss', info_loss.item())
                    self.averager_meters.update('fake_wid_loss', fake_wid_loss.item())
                    self.averager_meters.update('fdl_loss', fdl_loss.item())
                    self.averager_meters.update('kl_loss', kl_loss.item())
                    self.optimizers.G.step()

                if iter_count % self.opt.training.print_iter_val == 0:
                    meter_vals = self.averager_meters.eval_all()
                    self.averager_meters.reset_all()
                    info = "[%3d|%3d]-[%4d|%4d] G:%.4f D-fake:%.4f D-real:%.4f " \
                            "HF-fake:%.4f HF-real:%.4f CTC-fake:%.4f CTC-real:%.4f " \
                           "Wid-fake:%.4f Wid-real:%.4f Recn-z:%.4f FDL:%.4f Kl:%.4f" \
                           % (epoch, self.opt.training.epochs,
                              iter_count % len(self.train_loader), len(self.train_loader),
                              meter_vals['adv_loss'],
                              meter_vals['fake_disc_loss'], meter_vals['real_disc_loss'],
                              meter_vals['hf_fake_disc_loss'], meter_vals['hf_real_disc_loss'],
                              meter_vals['fake_ctc_loss'], meter_vals['real_ctc_loss'],
                              meter_vals['fake_wid_loss'], meter_vals['real_wid_loss'],
                              meter_vals['info_loss'], meter_vals['fdl_loss'], meter_vals['kl_loss'])
                    self.print(info)

                    if self.writer:
                        for key, val in meter_vals.items():
                            self.writer.add_scalar('loss/%s' % key, val, iter_count + 1)

                if (iter_count + 1) % self.opt.training.sample_iter_val == 0:
                    if not (self.logger and self.writer):
                        self.create_logger()

                    sample_root = os.path.join(self.log_root, self.opt.training.sample_dir)
                    if not os.path.exists(sample_root):
                        os.makedirs(sample_root)
                    self.sample_images(iter_count + 1)

                iter_count += 1

            if epoch:
                ckpt_root = os.path.join(self.log_root, self.opt.training.ckpt_dir)
                if not os.path.exists(ckpt_root):
                    os.makedirs(ckpt_root)

                self.save('last', epoch)
                if epoch >= self.opt.training.start_save_epoch_val and \
                        epoch % self.opt.training.save_epoch_val == 0:
                    self.print('Calculate FID_KID')
                    scores = self.validate()
                    fid, kid = scores['FID'], scores['KID']
                    self.print('FID:{} KID:{}'.format(fid, kid))

                    if kid < best_kid:
                        best_kid = kid
                        self.save('best', epoch, KID=kid, FID=fid)
                    if self.writer:
                        self.writer.add_scalar('valid/FID', fid, epoch)
                        self.writer.add_scalar('valid/KID', kid, epoch)

            for scheduler in self.lr_schedulers.values():
                scheduler.step(epoch)

    def sample_images(self, iteration_done=0):
        self.set_mode('eval')

        device = self.device
        batchA = next(iter(self.tst_loader))
        batchB = next(iter(self.tst_loader2))
        batch = Hdf5Dataset.merge_batch(batchA, batchB, device)
        imgs, img_lens, lbs, lb_lens, wids = batch

        real_imgs, real_img_lens = imgs.to(device), img_lens.to(device)
        real_lbs, real_lb_lens = lbs.to(device), lb_lens.to(device)

        with torch.no_grad():
            self.eval_z.sample_()
            recn_imgs = None
            if 'E' in self.models:
                enc_styles = self.models.E(real_imgs, real_img_lens, self.models.S)
                noises = torch.randn((real_imgs.size(0), self.opt.GenModel.style_dim
                                      - self.opt.EncModel.style_dim)).float().to(device)
                enc_z = torch.cat([noises, enc_styles], dim=-1)
                recn_imgs = self.models.G(enc_z, real_lbs, real_lb_lens)

            fake_real_imgs = self.models.G(self.eval_z, real_lbs, real_lb_lens)

            self.eval_y.sample_()
            sampled_words = idx_to_words(self.eval_y, self.lexicon, self.opt.training.capitalize_ratio)
            sampled_words[-2] = sampled_words[-1]
            fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
            fake_lbs, fake_lb_lens = fake_lbs.to(device), fake_lb_lens.to(device)
            fake_imgs = self.models.G(self.eval_z, fake_lbs, fake_lb_lens)

            max_img_len = max([real_imgs.size(-1), fake_real_imgs.size(-1), fake_imgs.size(-1)])
            img_shape = [real_imgs.size(2), max_img_len, real_imgs.size(1)]

            real_imgs = F.pad(real_imgs, [0, max_img_len - real_imgs.size(-1), 0, 0], value=-1.)
            fake_real_imgs = F.pad(fake_real_imgs, [0, max_img_len - fake_real_imgs.size(-1), 0, 0], value=-1.)
            fake_imgs = F.pad(fake_imgs, [0, max_img_len - fake_imgs.size(-1), 0, 0], value=-1.)
            recn_imgs = F.pad(recn_imgs, [0, max_img_len - recn_imgs.size(-1), 0, 0], value=-1.) \
                        if recn_imgs is not None else None

            real_words = self.label_converter.decode(real_lbs, real_lb_lens)
            real_labels = words_to_images(real_words, *img_shape)
            rand_labels = words_to_images(sampled_words, *img_shape)

            try:
                sample_img_list = [real_labels.cpu(), real_imgs.cpu(), fake_real_imgs.cpu(),
                                   fake_imgs.cpu(), rand_labels.cpu()]
                if recn_imgs is not None:
                    sample_img_list.insert(2, recn_imgs.cpu())
                sample_imgs = torch.cat(sample_img_list, dim=2).repeat(1, 3, 1, 1)
                res_img = draw_image(1 - sample_imgs.data, nrow=self.opt.training.sample_nrow, normalize=True)
                save_path = os.path.join(self.log_root, self.opt.training.sample_dir,
                                         'iter_{}.png'.format(iteration_done))
                im = Image.fromarray(res_img)
                im.save(save_path)
                if self.writer:
                    self.writer.add_image('Image', res_img.transpose((2, 0, 1)), iteration_done)
            except RuntimeError as e:
                print(e)

    def image_generator(self, source_dloader, style_guided=True):
        device = self.device

        with torch.no_grad():
            for style_imgs, style_img_lens, style_lbs, style_lb_lens, style_wids in source_dloader:
                content_lbs, content_lb_lens = style_lbs.to(device), style_lb_lens.to(device)

                if style_guided:
                    enc_styles = self.models.E(style_imgs.to(device), style_img_lens.to(device),
                                               self.models.S)
                    noises = torch.randn((style_imgs.size(0), self.opt.GenModel.style_dim
                                          - self.opt.EncModel.style_dim)).float().to(device)
                    enc_z = torch.cat([noises, enc_styles], dim=-1)
                else:
                    enc_z = torch.randn(style_imgs.size(0), self.opt.GenModel.style_dim).to(device)

                fake_imgs = self.models.G(enc_z, content_lbs.long(), content_lb_lens.long())
                fake_img_lens = content_lb_lens * self.opt.char_width
                yield fake_imgs, fake_img_lens, content_lbs, content_lb_lens, style_wids.to(device)


    def validate(self, guided=True):
        self.set_mode('eval')
        dset_name = self.opt.valid.dset_name if self.opt.valid.dset_name \
                    else self.opt.dataset
        dset = get_dataset(dset_name, self.opt.valid.dset_split)
        dloader = DataLoader(
            dset,
            collate_fn=self.collect_fn,
            batch_size=self.opt.valid.batch_size,
            shuffle=False,
            num_workers=4
        )
        # style images are resized
        source_dloader = DataLoader(
            get_dataset(self.opt.valid.dset_name.strip('_org'), self.opt.valid.dset_split),
            collate_fn=self.collect_fn,
            batch_size=self.opt.valid.batch_size,
            shuffle=False,
            num_workers=4
        )
        generator = self.image_generator(source_dloader, guided)
        fid_kid = calculate_kid_fid(self.opt.valid, dloader, generator, self.max_valid_image_width, self.device)
        return fid_kid

    def eval_interp(self):
        self.set_mode('eval')

        with torch.no_grad():
            interp_num = self.opt.test.interp_num
            nrow, ncol = 1, interp_num
            while True:
                text = input('input text: ')
                if len(text) == 0:
                    break

                fake_lbs = self.label_converter.encode(text)
                fake_lbs = torch.LongTensor(fake_lbs)
                fake_lb_lens = torch.IntTensor([len(text)])

                style0 = torch.randn((1, self.opt.GenModel.style_dim))
                style1 = torch.randn(style0.size())
                noise = torch.randn((1, self.noise_dim)).repeat(interp_num, 1).to(self.device)

                styles = [torch.lerp(style0, style1, i / (interp_num - 1)) for i in range(interp_num)]
                styles = torch.cat(styles, dim=0).float().to(self.device)
                styles = torch.cat([noise, styles], dim=1).to(self.device)

                fake_lbs, fake_lb_lens = fake_lbs.repeat(nrow * ncol, 1).to(self.device),\
                                         fake_lb_lens.repeat(nrow * ncol).to(self.device)
                gen_imgs = self.models.G(styles, fake_lbs, fake_lb_lens)
                gen_imgs = (1 - gen_imgs).squeeze().cpu().numpy() * 127
                plt.figure()
                for i in range(nrow * ncol):
                    plt.subplot(nrow, ncol, i + 1)
                    plt.imshow(gen_imgs[i], cmap='gray')
                    plt.axis('off')
                plt.tight_layout()
                plt.show()

    def image_generator_custom(self, source_dloader, style_guided=False, use_sampled_words=True):
        device = self.device
        opt = self.opt
        
        if use_sampled_words:
            max_batch_size = self.opt.valid.batch_size
        
        with torch.no_grad():
            for (
                style_imgs,
                style_img_lens,
                style_lbs,
                style_lb_lens,
                style_wids,
            ) in source_dloader:
                batch_size = style_imgs.size(0)
                
                # Get style information
                if style_guided:
                    enc_styles = self.models.E(
                        style_imgs.to(device),
                        style_img_lens.to(device),
                        self.models.S,
                    )
                    noises = torch.randn((batch_size, self.opt.GenModel.style_dim - self.opt.EncModel.style_dim)).float().to(device)
                    enc_z = torch.cat([noises, enc_styles], dim=-1)
                else:
                    enc_z = torch.randn(batch_size, self.opt.GenModel.style_dim).to(device)
                
                if use_sampled_words:
                    self.temp_y_dist.sample_()
                    sampled_words = idx_to_words(
                        self.temp_y_dist[:batch_size],
                        self.lexicon, 
                        self.opt.training.capitalize_ratio
                    )
                    
                    fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
                    content_lbs = torch.LongTensor(fake_lbs).to(device)
                    content_lb_lens = torch.IntTensor(fake_lb_lens).to(device)
                else:
                    content_lbs, content_lb_lens = style_lbs.to(device), style_lb_lens.to(device)
    
                # Generate images
                fake_imgs = self.models.G(enc_z, content_lbs.long(), content_lb_lens.long())
                fake_img_lens = content_lb_lens * self.opt.char_width
                
                yield fake_imgs, fake_img_lens, content_lbs, content_lb_lens, style_wids.to(device)

    def image_generator_custom_CER(self, source_dloader, style_guided=False, use_sampled_words=True):
        device = self.device
        opt = self.opt
    
        with torch.no_grad():
            for (
                style_imgs,
                style_img_lens,
                style_lbs,
                style_lb_lens,
                style_wids,
            ) in source_dloader:
                batch_size = style_imgs.size(0)
    
                # Get style information
                if style_guided:
                    enc_styles = self.models.E(
                        style_imgs.to(device),
                        style_img_lens.to(device),
                        self.models.S,
                    )
                    noises = torch.randn((batch_size, self.opt.GenModel.style_dim - self.opt.EncModel.style_dim)).float().to(device)
                    enc_z = torch.cat([noises, enc_styles], dim=-1)
                else:
                    enc_z = torch.randn(batch_size, self.opt.GenModel.style_dim).to(device)

                if use_sampled_words:
                    self.temp_y_dist.sample_()
                    sampled_words = idx_to_words(
                        self.temp_y_dist[:batch_size],
                        self.lexicon, 
                        self.opt.training.capitalize_ratio
                    )
                    fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
                    content_lbs = torch.LongTensor(fake_lbs).to(device)
                    content_lb_lens = torch.IntTensor(fake_lb_lens).to(device)
                else:
                    content_lbs, content_lb_lens = style_lbs.to(device), style_lb_lens.to(device)
    
                fake_imgs = self.models.G(enc_z, content_lbs.long(), content_lb_lens.long())
                fake_img_lens = content_lb_lens * self.opt.char_width
    
                yield fake_imgs, fake_img_lens, content_lbs, content_lb_lens, style_wids.to(device)

    def gen_random_images(self, guided=True, total=25000):
        import cv2
        from tqdm import tqdm
        self.set_mode("eval")
        dset_name = (
            self.opt.valid.dset_name if self.opt.valid.dset_name else self.opt.dataset
        )
        dset = get_dataset(dset_name, self.opt.valid.dset_split)
        dloader = DataLoader(
            dset,
            collate_fn=self.collect_fn,
            batch_size=self.opt.valid.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=True
        )
        def create_source_dloader():
            return DataLoader(
                get_dataset(self.opt.valid.dset_name, self.opt.valid.dset_split),
                collate_fn=self.collect_fn,
                batch_size=self.opt.valid.batch_size,
                shuffle=False,
                num_workers=4,
                drop_last=True
            )
    
        fake_base = os.path.join("/kaggle/working/", "test-fake")
        os.makedirs(fake_base, exist_ok=True)
    
        # Prepare the random distribution
        max_batch_size = self.opt.valid.batch_size
        self.temp_y_dist = prepare_y_dist(max_batch_size, len(self.lexicon), self.device, seed=self.opt.seed)
    
        idx = 0
        with tqdm(total=total, desc="Generating images") as pbar:
            while idx < total:
                source_dloader = create_source_dloader()
                generator2 = self.image_generator_custom_CER(source_dloader, guided, use_sampled_words=True)
                try:
                    for batch in generator2:
                        if idx >= total:
                            break
                        imgs, img_lens, lb, lb_len, w = batch
                        lb_len = lb_len * self.opt.char_width
    
                        for i in range(imgs.shape[0]):
                            if idx >= total:
                                break
                            lbs = self.label_converter.decode(lb[i])
                            image = imgs[i, :, :, :lb_len[i]]
                            image = 255 * ((image[0] + 1) / 2)
                            image = image.cpu().numpy()
                            cv2.imwrite(
                                "/kaggle/working/test-fake/fw" + str(idx) + ".png",
                                image,
                            )
                            with open(
                                "/kaggle/working/test-fake/fw.txt",
                                "a",
                            ) as f:
                                label_str = ''.join(lbs[:lb_len[i] // self.opt.char_width])
                                f.write(f"fw{idx}.png\t{label_str}\n")
                            idx += 1
                            pbar.update(1)
                except StopIteration:
                    continue 

    def gen_fakes(self, guided=True, use_random_lexicon=False):
        import json
        import cv2
        self.set_mode('eval')
        
        real_root = os.path.join(self.log_root, 'reals_images')
        fake_root = os.path.join(self.log_root, 'fakes_images')
        os.makedirs(real_root, exist_ok=True)
        os.makedirs(fake_root, exist_ok=True)
        
        dset_name = self.opt.valid.dset_name if self.opt.valid.dset_name else self.opt.dataset
        dset = get_dataset(dset_name, self.opt.valid.dset_split)
        source_dset_name = dset_name.strip('_org') if '_org' in dset_name else dset_name
        
        dloader = DataLoader(
            dset,
            collate_fn=self.collect_fn,
            batch_size=self.opt.valid.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        source_dloader = DataLoader(
            get_dataset(source_dset_name, self.opt.valid.dset_split),
            collate_fn=self.collect_fn,
            batch_size=self.opt.valid.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        real_count = 0
        fake_count = 0
        author_count = {}
        real_transcriptions = {}
        
        # Process real images
        for real_imgs, real_img_lens, real_lbs, real_lb_lens, wids in tqdm(dloader, desc="Saving real images"):
            real_imgs = real_imgs.to(self.device)
            batch_size = real_imgs.size(0)
            
            real_texts = self.label_converter.decode(real_lbs, real_lb_lens)
            
            for i in range(batch_size):
                wid = wids[i].item()
                img_len = real_img_lens[i].item()
                text = real_texts[i]
                
                author_dir = os.path.join(real_root, f'{wid}')
                os.makedirs(author_dir, exist_ok=True)
                
                if wid not in author_count:
                    author_count[wid] = 0
                
                image = real_imgs[i, :, :, :img_len]
                image = 255 * ((image[0] + 1) / 2)

                image = pad(image, img_len, lenlb=len(text))
                
                img_filename = f'{author_count[wid]:04d}.png'
                img_path = os.path.join(author_dir, img_filename)
                cv2.imwrite(img_path, image)
                
                rel_path = f"{wid}/{img_filename}"
                real_transcriptions[rel_path] = text
                
                author_count[wid] += 1
                real_count += 1
        
        with open(os.path.join(real_root, 'transcriptions.json'), 'w') as f:
            json.dump(real_transcriptions, f, indent=2)
        
        author_count = {}
        fake_transcriptions = {}
        
        if use_random_lexicon:
            max_batch_size = self.opt.valid.batch_size
            self.temp_y_dist = prepare_y_dist(max_batch_size, len(self.lexicon), self.device, seed=self.opt.seed)
            generator = self.image_generator_custom(source_dloader, guided, use_sampled_words=True)
        else:
            generator = self.image_generator(source_dloader, guided)
            
            
        # Process fake images
        for fake_imgs, fake_img_lens, fake_lbs, fake_lb_lens, wids in tqdm(generator, desc="Saving fake images"):
            batch_size = fake_imgs.size(0)
            
            fake_texts = self.label_converter.decode(fake_lbs, fake_lb_lens)
            
            for i in range(batch_size):
                wid = wids[i].item()
                img_len = fake_img_lens[i].item()
                text = fake_texts[i]
                
                author_dir = os.path.join(fake_root, f'{wid}')
                os.makedirs(author_dir, exist_ok=True)
                
                if wid not in author_count:
                    author_count[wid] = 0
                
                image = fake_imgs[i, :, :, :img_len]
                image = 255 * ((image[0] + 1) / 2)

                image = pad(image, img_len, lenlb=len(text))
                
                img_filename = f'{author_count[wid]:04d}.png'
                img_path = os.path.join(author_dir, img_filename)
                cv2.imwrite(img_path, image)
                
                rel_path = f"{wid}/{img_filename}"
                fake_transcriptions[rel_path] = text
                
                author_count[wid] += 1
                fake_count += 1
        
        with open(os.path.join(fake_root, 'transcriptions.json'), 'w') as f:
            json.dump(fake_transcriptions, f, indent=2)
        
        sampling_mode = "random lexicon" if use_random_lexicon else "original text"
        self.print(f"Saved {real_count} real images from {len(os.listdir(real_root))-1} authors")  # -1 for transcriptions.json
        self.print(f"Saved {fake_count} fake images from {len(os.listdir(fake_root))-1} authors using {sampling_mode}")
        self.print(f"Created transcription files for CER calculation")
        
        return real_root, fake_root

    def _preprocess_sentences(self, sentences):
        
        if sentences is None:
            return [["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]]
        
        processed_sentences = []
        for sentence in sentences:
            if isinstance(sentence, str):
                # Split string by spaces and filter out empty strings
                words = [word.strip() for word in sentence.split(' ') if word.strip()]
                processed_sentences.append(words)
            elif isinstance(sentence, list):
                # Already a list of words
                processed_sentences.append(sentence)
            else:
                raise ValueError(f"Sentence must be a string or list of words, got {type(sentence)}")
        
        return processed_sentences

    def save_images_from_sentence(self, save_root=None, sentences=None):
        # Default sentence if none provided
        if sentences is None:
            sentences = ["The quick brown fox jumps over the lazy dog"]

        sentences = self._preprocess_sentences(sentences)
    
        self.set_mode('eval')
        device = self.device
    
        if save_root is None:
            save_root = os.path.join(self.log_root, 'sentence_images')
        os.makedirs(save_root, exist_ok=True)
    
        dataset = get_dataset(
            self.opt.valid.dset_name,
            self.opt.valid.dset_split
        )
        dataloader = DataLoader(
            dataset,
            batch_size=5,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collect_fn,
            drop_last=False
        )
    
        wid_sample_counts = {}
        total_generated = 0
        
        for batch in tqdm(dataloader, desc="Processing writer styles"):
            imgs, img_lens, lbs, lb_lens, wids = batch
            style_imgs, style_img_lens = imgs.to(device), img_lens.to(device)
            
            with torch.no_grad():
                if 'E' in self.models:
                    enc_styles = self.models.E(style_imgs, style_img_lens, self.models.S)
                    noises = torch.randn((style_imgs.size(0), self.opt.GenModel.style_dim
                                          - self.opt.EncModel.style_dim)).float().to(device)
                    enc_z = torch.cat([noises, enc_styles], dim=-1)
                else:
                    enc_z = torch.randn(style_imgs.size(0), self.opt.GenModel.style_dim).to(device)
                
                for sentence in sentences:
                    batch_words = []
                    for word in sentence:
                        batch_words.extend([word] * style_imgs.size(0))
                    
                    fake_lbs, fake_lb_lens = self.label_converter.encode(batch_words)
                    fake_lbs, fake_lb_lens = torch.LongTensor(fake_lbs).to(device), torch.IntTensor(fake_lb_lens).to(device)
                    
                    for i in range(style_imgs.size(0)):
                        wid = wids[i].item()
                        
                        if wid not in wid_sample_counts:
                            wid_sample_counts[wid] = 0
                        
                        wid_dir = os.path.join(save_root, f'wid{wid}')
                        os.makedirs(wid_dir, exist_ok=True)
                        
                        fake_imgs_list = []
                        
                        fake_imgs_list.append(style_imgs[i:i+1][:, :, :, :style_img_lens[i]])
                        
                        padding = torch.ones(
                            1,
                            style_imgs.size(1),
                            style_imgs.size(2),
                            16,
                        ).to(device)
                        fake_imgs_list.append(padding)
                        
                        for j in range(len(sentence)):
                            word_idx = i + j * style_imgs.size(0)
                            word_lbs = fake_lbs[word_idx:word_idx+1]
                            word_lb_lens = fake_lb_lens[word_idx:word_idx+1]
                            
                            fake_img = self.models.G(enc_z[i:i+1], word_lbs, word_lb_lens)
                            fake_img = fake_img[:, :, :, :word_lb_lens * self.opt.char_width]
                            fake_imgs_list.append(fake_img)
                            
                            if j < len(sentence) - 1:
                                padding = torch.ones(
                                    1,
                                    fake_img.size(1),
                                    fake_img.size(2),
                                    16,
                                ).to(device)
                                fake_imgs_list.append(padding)
                        
                        fake_imgs = torch.cat(fake_imgs_list, dim=3)
                        
                        img = fake_imgs[0][0]  
                        img = 255 * ((img + 1) / 2)  
                        img = img.cpu().numpy()
                        
                        sentence_str = '_'.join(sentence)
                        path = os.path.join(wid_dir, f'{wid_sample_counts[wid]:04d}_{sentence_str}.png')
                        cv2.imwrite(path, img)
                        
                        # Update counters
                        wid_sample_counts[wid] += 1
                        total_generated += 1
    
        self.print(f"Generated {total_generated} images across {len(wid_sample_counts)} writer IDs")
        self.print(f"Images saved to {save_root}")
        
        wid_counts_str = ", ".join([f"wid{wid}: {count}" for wid, count in wid_sample_counts.items()])
        self.print(f"Sample counts per writer ID: {wid_counts_str}")
        
        return save_root

    def save_images_from_reference_labels(self, save_root=None, max_samples_per_writer=None):
        
        self.set_mode('eval')
        device = self.device
    
        if save_root is None:
            save_root = os.path.join(self.log_root, 'reconstructed_images_by_wid')
        os.makedirs(save_root, exist_ok=True)
    
        dataset = get_dataset(
            self.opt.valid.dset_name,
            self.opt.valid.dset_split
        )
        dataloader = DataLoader(
            dataset,
            batch_size=5,
            shuffle=False,
            num_workers=4,
            collate_fn=self.collect_fn,
            drop_last=False
        )
    
        wid_sample_counts = {}
        total_generated = 0
        
        for batch in tqdm(dataloader, desc="Reconstructing images from reference labels"):
            imgs, img_lens, lbs, lb_lens, wids = batch
            style_imgs, style_img_lens = imgs.to(device), img_lens.to(device)
            lbs, lb_lens = lbs.to(device), lb_lens.to(device)
            
            original_labels = lbs.cpu().numpy()
            original_label_lens = lb_lens.cpu().numpy()
            
            with torch.no_grad():
                if 'E' in self.models:
                    enc_styles = self.models.E(style_imgs, style_img_lens, self.models.S)
                    noises = torch.randn((style_imgs.size(0), self.opt.GenModel.style_dim
                                          - self.opt.EncModel.style_dim)).float().to(device)
                    enc_z = torch.cat([noises, enc_styles], dim=-1)
                else:
                    enc_z = torch.randn(style_imgs.size(0), self.opt.GenModel.style_dim).to(device)
                
                for i in range(style_imgs.size(0)):
                    wid = wids[i].item()
                    
                    if max_samples_per_writer is not None:
                        if wid in wid_sample_counts and wid_sample_counts[wid] >= max_samples_per_writer:
                            continue
                    
                    if wid not in wid_sample_counts:
                        wid_sample_counts[wid] = 0
                    
                    wid_dir = os.path.join(save_root, f'wid{wid}')
                    os.makedirs(wid_dir, exist_ok=True)
                    
                    original_label = original_labels[i][:original_label_lens[i]]
                    decoded_text = self.label_converter.decode(original_label)
                    
                    img_list = []
                    
                    ref_img = style_imgs[i:i+1][:, :, :, :style_img_lens[i]]
                    img_list.append(ref_img)
                    
                    padding = torch.ones(1, ref_img.size(1), ref_img.size(2), 16).to(device)
                    img_list.append(padding)
                    
                    fake_img = self.models.G(enc_z[i:i+1], lbs[i:i+1], lb_lens[i:i+1])
                    fake_img = fake_img[:, :, :, :lb_lens[i] * self.opt.char_width]
                    img_list.append(fake_img)
                    
                    combined_img = torch.cat(img_list, dim=3)
                    
                    img = combined_img[0][0]  
                    img = 255 * ((img + 1) / 2)  
                    img = img.cpu().numpy()
                    
                    clean_text = "".join(c for c in decoded_text if c.isalnum() or c in (' ', '-', '_')).strip()
                    clean_text = clean_text.replace(' ', '_')
                    path = os.path.join(wid_dir, f'{wid_sample_counts[wid]:04d}_ref_vs_gen_{clean_text}.png')
                    cv2.imwrite(path, img)
                    
                    # Update counters
                    wid_sample_counts[wid] += 1
                    total_generated += 1
    
        self.print(f"Generated {total_generated} reconstructed images across {len(wid_sample_counts)} writer IDs")
        self.print(f"Images saved to {save_root}")
        
        # Print sample counts per writer ID
        wid_counts_str = ", ".join([f"wid{wid}: {count}" for wid, count in wid_sample_counts.items()])
        self.print(f"Sample counts per writer ID: {wid_counts_str}")
        
        return save_root

    def save_paragraph(self, save_root=None, sentences=None, words_per_line=10):
    
        # Default sentence if none provided
        if sentences is None:
            sentences = ["The quick brown fox jumps over the lazy dog"]
    
        sentences = self._preprocess_sentences(sentences)
    
        self.set_mode('eval')
        device = self.device
    
        # Output root directory
        if save_root is None:
            save_root = os.path.join(self.log_root, 'sentence_images_by_wid')
        os.makedirs(save_root, exist_ok=True)
    
        dataset = get_dataset(
            self.opt.valid.dset_name,
            self.opt.valid.dset_split
        )
        dataloader = DataLoader(
            dataset,
            batch_size=5,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collect_fn,
            drop_last=False
        )
    
        wid_sample_counts = {}
        total_generated = 0
        
        for batch in tqdm(dataloader, desc="Processing writer styles"):
            imgs, img_lens, lbs, lb_lens, wids = batch
            style_imgs, style_img_lens = imgs.to(device), img_lens.to(device)
            
            with torch.no_grad():
                if 'E' in self.models:
                    enc_styles = self.models.E(style_imgs, style_img_lens, self.models.S)
                    noises = torch.randn((style_imgs.size(0), self.opt.GenModel.style_dim
                                          - self.opt.EncModel.style_dim)).float().to(device)
                    enc_z = torch.cat([noises, enc_styles], dim=-1)
                else:
                    enc_z = torch.randn(style_imgs.size(0), self.opt.GenModel.style_dim).to(device)
                
                for sentence in sentences:
                    lines = []
                    for i in range(0, len(sentence), words_per_line):
                        line = sentence[i:i + words_per_line]
                        lines.append(line)
                    
                    batch_words = []
                    for line in lines:
                        for word in line:
                            batch_words.extend([word] * style_imgs.size(0))
                    
                    fake_lbs, fake_lb_lens = self.label_converter.encode(batch_words)
                    fake_lbs, fake_lb_lens = torch.LongTensor(fake_lbs).to(device), torch.IntTensor(fake_lb_lens).to(device)
                    
                    for i in range(style_imgs.size(0)):
                        wid = wids[i].item()
                        
                        if wid not in wid_sample_counts:
                            wid_sample_counts[wid] = 0
                        
                        wid_dir = os.path.join(save_root, f'wid{wid}')
                        os.makedirs(wid_dir, exist_ok=True)
                        
                        paragraph_lines = []
                        
                        ref_img = style_imgs[i:i+1][:, :, :, :style_img_lens[i]]
                        paragraph_lines.append(ref_img)
                        
                        ref_padding = torch.ones(
                            1,
                            style_imgs.size(1),
                            style_imgs.size(2),
                            16,
                        ).to(device)
                        paragraph_lines.append(ref_padding)
                        
                        word_idx_offset = 0
                        for line_idx, line in enumerate(lines):
                            line_imgs_list = []
                            
                            for j, word in enumerate(line):
                                word_idx = i + word_idx_offset * style_imgs.size(0)
                                word_lbs = fake_lbs[word_idx:word_idx+1]
                                word_lb_lens = fake_lb_lens[word_idx:word_idx+1]
                                
                                fake_img = self.models.G(enc_z[i:i+1], word_lbs, word_lb_lens)
                                fake_img = fake_img[:, :, :, :word_lb_lens * self.opt.char_width]
                                line_imgs_list.append(fake_img)
                                
                                if j < len(line) - 1:
                                    word_padding = torch.ones(
                                        1,
                                        fake_img.size(1),
                                        fake_img.size(2),
                                        16,
                                    ).to(device)
                                    line_imgs_list.append(word_padding)
                                
                                word_idx_offset += 1
                            
                            line_img = torch.cat(line_imgs_list, dim=3)
                            paragraph_lines.append(line_img)
                            
                            if line_idx < len(lines) - 1:
                                line_padding = torch.ones(
                                    1,
                                    line_img.size(1),
                                    8,
                                    line_img.size(3),
                                ).to(device)
                                paragraph_lines.append(line_padding)
                        
                        max_width = max([line.size(3) for line in paragraph_lines])
                        
                        padded_lines = []
                        for line in paragraph_lines:
                            if line.size(3) < max_width:
                                right_padding = torch.ones(
                                    line.size(0),
                                    line.size(1),
                                    line.size(2),
                                    max_width - line.size(3)
                                ).to(device)
                                padded_line = torch.cat([line, right_padding], dim=3)
                            else:
                                padded_line = line
                            padded_lines.append(padded_line)
                        
                        paragraph_img = torch.cat(padded_lines, dim=2)
                        
                        
                        img = paragraph_img[0][0]  
                        img = 255 * ((img + 1) / 2)  
                        img = img.cpu().numpy()
                        
                        sentence_str = '_'.join(sentence)
                        if len(sentence_str) > 100:
                            sentence_str = sentence_str[:100] + "..."
                        path = os.path.join(wid_dir, f'{wid_sample_counts[wid]:04d}_{words_per_line}wpl_{sentence_str}.png')
                        cv2.imwrite(path, img)
                        
                        wid_sample_counts[wid] += 1
                        total_generated += 1
    
        self.print(f"Generated {total_generated} paragraph images across {len(wid_sample_counts)} writer IDs")
        self.print(f"Images saved to {save_root} with {words_per_line} words per line")
        
        # Print sample counts per writer ID
        wid_counts_str = ", ".join([f"wid{wid}: {count}" for wid, count in wid_sample_counts.items()])
        self.print(f"Sample counts per writer ID: {wid_counts_str}")
        
        return save_root