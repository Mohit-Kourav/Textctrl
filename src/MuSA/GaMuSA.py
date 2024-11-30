import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from src.module.abinet.modules.model_vision import BaseVision
import torchvision.transforms as T
from omegaconf import OmegaConf
from src.module.abinet import CharsetMapper
import os
from PIL import Image
import pickle
from torchvision.transforms import ToPILImage


class GaMuSA:
    def __init__(self, model, monitor_cfg):
        self.model = model
        self.scheduler = model.noise_scheduler
        self.device = model.device
        self.unet = model.unet
        self.vae = model.vae
        self.control_model = model.control_model
        self.text_encoder = model.text_encoder
        self.NORMALIZER = model.NORMALIZER
        monitor_cfg = OmegaConf.create(monitor_cfg)
        self.monitor = BaseVision(monitor_cfg).to(self.device)
        self.charset = CharsetMapper(filename=monitor_cfg.charset_path)
        self.max_length = monitor_cfg.max_length + 1

    def next_step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            x: torch.FloatTensor,
            eta=0.,
            verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        x_next = alpha_prod_t_next ** 0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            assert image.dim() == 4, print("input dims should be 4 !")
            latents = self.vae.encode(image.to(self.device)).latent_dist.sample()
            latents = latents * self.NORMALIZER
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / self.NORMALIZER * latents.detach()
        # print("decoder latent -=-=--=-=-=-=-=-=-=-==-=-= ", latents.shape)
        image = self.model.vae.decode(latents).sample
        if return_type == 'np':
            image = image.clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    def latent2image_grad(self, latents):
        latents = 1 / self.NORMALIZER * latents
        image = self.vae.decode(latents).sample
        return image

    @torch.no_grad()
    def inversion(
            self,
            image: torch.Tensor,
            hint: torch.Tensor,
            cond,
            num_inference_steps=50,
            guidance_scale=7.5,
            eta=0.0,
            return_intermediates=False,
            **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        # print("cond -=-=-=-=-=-==-", cond)
        assert image.shape[0] == len(cond), print("Unequal batch size for image and cond.")
        assert image.shape[0] == hint.shape[0], print("Unequal batch size for image and hint.")

        cond_embeddings = self.model.get_text_conditioning(cond)
        if guidance_scale > 1.:
            uncond = [""] * len(cond)
            uncond_embeddings = self.model.get_text_conditioning(uncond)
            text_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
        else:
            text_embeddings = cond_embeddings

        latents = self.image2latent(image)
        start_latents = latents


        self.scheduler.set_timesteps(num_inference_steps)
        latents_list = [latents]
        pred_x0_list = [latents]
        print("shape of hint image inversion ***************", hint.shape)
        
        for i, t in enumerate(reversed(self.scheduler.timesteps)):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
                hint_input = torch.cat([hint] * 2)
            else:
                model_inputs = latents
                hint_input = hint
            control_input = self.control_model(hint_input, model_inputs, t, text_embeddings)
            print("shape of control_input inversion ***************", control_input[0].shape)
            output_path = "latent_features.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(control_input[0], f)
            print(f"Tensor saved at {output_path}")
            noise_pred = self.unet(
                x=model_inputs,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                control=control_input
            ).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            
            save_dir = "latent_images_inversion"
            os.makedirs(save_dir, exist_ok=True)
            
            # Convert latents to images
            glyph_inputs = self.latent2image(latents, return_type="pt")
            # print(f"Glyph inputs shape before resizing: {glyph_inputs.shape}")
            
            # Check and unsqueeze if necessary
            if glyph_inputs.dim() == 3:  # Already (b, c, h, w)
                glyph_inputs_1 = glyph_inputs.unsqueeze(0)  # Adds an extra batch dim
            else:
                glyph_inputs_1 = glyph_inputs
            
            # Resize the tensor
            glyph_resize = T.Resize([32, 128])
            glyph_inputs_resized = torch.stack([glyph_resize(img) for img in glyph_inputs])  # Apply resize per batch
            
            # Save each image in the batch
            for idx, image_tensor in enumerate(glyph_inputs_resized):
                # Convert tensor to PIL Image
                image_pil = T.ToPILImage()(image_tensor.cpu())
                
                # Save the image
                image_path = os.path.join(save_dir, f"image_{i+1}_{idx}.png")
                image_pil.save(image_path)
            
            
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)
        if return_intermediates:
            return latents, latents_list
        return latents, start_latents

    @torch.no_grad()
    def __call__(
            self,
            hint,
            cond,
            start_step=24,
            start_layer=10,
            batch_size=1,
            height=256,
            width=256,
            num_inference_steps=50,
            guidance_scale=2,
            eta=0.0,
            latents=None,
            unconditioning=None,
            neg_prompt=None,
            ref_intermediate_latents=None,
            return_intermediates=False,
            enable_GaMuSA=False,
            **kwds):
        # print("cond -=-=-==-=-=-=-=-=-=-==-==-=-=> " , cond)
        # print("hint -=-=-=-==-=--=-=-=-=-=-=-=-=-=-" , hint.shape)
        
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
        # Iterate over the batch (first dimension of tensor)
        for idx, img_tensor in enumerate(hint):
            # Normalize tensor to range [0, 255]
            img_tensor = img_tensor.permute(1, 2, 0)  # Change shape to HWC (height, width, channels)
            img_tensor = (img_tensor * 255).byte()  # Scale values and convert to byte format

            # Convert to PIL Image
            img = Image.fromarray(img_tensor.cpu().numpy())

            # Save the image
            img.save(os.path.join(output_dir, f"image_{idx}.png"))
        
        
        cond_embeddings = self.model.get_text_conditioning(cond)
        # print("condition embeddings -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=->", cond_embeddings.shape)
        if guidance_scale > 1.:
            uncond = [""] * len(cond)
            uncond_embeddings = self.model.get_text_conditioning(uncond)
            text_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
        else:
            text_embeddings = cond_embeddings

        batch_size = len(cond)
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(latents_shape, device=self.device)
        else:
            pass

        self.scheduler.set_timesteps(num_inference_steps)
        latents_list = [latents]
        pred_x0_list = [latents]

        gt_ids, _ = prepare_label(cond, self.charset, self.max_length, self.device)
        # print(gt_ids.shape)
        from src.MuSA.utils import MuSA_TextCtrl
        from src.MuSA.utils import regiter_attention_editor_diffusers_Edit
        if enable_GaMuSA:
            controller = MuSA_TextCtrl(start_step, start_layer)
            regiter_attention_editor_diffusers_Edit(self.unet, controller)
            controller.start_ctrl()

        # print("-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        # c_m = self.control_model
        # # Display all parameters with their names
        # print("Parameters of the control model:")
        # for name, param in c_m.named_parameters():
        #     print(f"{name}: {param.size()}")

        # # Counting the total number of parameters
        # total_params = sum(p.numel() for p in c_m.parameters())
        # trainable_params = sum(p.numel() for p in c_m.parameters() if p.requires_grad)

        # print(f"\nTotal parameters in the control model: {total_params}")
        # print(f"Trainable parameters in the control model: {trainable_params}")
        # print("-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        
        
        output_folder = "hint_images"
        os.makedirs(output_folder, exist_ok=True)
        for i, t in enumerate(self.scheduler.timesteps):
            if ref_intermediate_latents is not None:
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
                hint_input = torch.cat([hint] * 2)
            else:
                model_inputs = latents
                hint_input = hint
                
            input_tensor = hint_input.to("cpu")

            # Folder to save the images
            

            # Transform to convert tensor to PIL image
            to_pil = ToPILImage()

            # Save each image
            for j in range(input_tensor.size(0)):  # Iterate over the batch dimension
                image_tensor = input_tensor[j]  # Shape: [3, 256, 256]
                image = to_pil(image_tensor)    # Convert to PIL Image
                image_path = os.path.join(output_folder, f"image_{i}_{j + 1}.png")  # Save as PNG
                image.save(image_path)  
                
            print("hint inpput in main ***************" , hint_input.shape)
            control_input = self.control_model(hint_input, model_inputs, t, text_embeddings)
            # print("length control_input **********************" , len(control_input))
            # for i, control_i in enumerate(control_input):
            #     print(f"control_input_{i}_shape **********************" , control_input[i].shape)
            # print("model_inputs_shape **********************" , model_inputs.shape)
            
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings])
            # print("text embeddings shape ********************", text_embeddings.shape)
            noise_pred = self.unet(
                x=model_inputs,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                control=control_input,
            ).sample
            # print("noise_pred -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=->" , noise_pred.shape)
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            latents, pred_x0 = self.step(noise_pred, t, latents)
            # print("latents -=-=-=-=-=-=-=-=-=-=--=--=->", latents.shape)
            # print("pred_x0 -=-=-=-=-=-=-=-=-=-=--=--=->", pred_x0.shape)
            
            
            if enable_GaMuSA:
                glyph_resize = T.transforms.Resize([32, 128])
                if (i + 1) % 5 == 0:
                    # Create directory to save images
                    save_dir = "saved_images"
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Convert latents to images
                    glyph_inputs = self.latent2image(latents, return_type="pt")
                    print(f"Glyph inputs shape before resizing: {glyph_inputs.shape}")
                    
                    # Check and unsqueeze if necessary
                    if glyph_inputs.dim() == 3:  # Already (b, c, h, w)
                        glyph_inputs_1 = glyph_inputs.unsqueeze(0)  # Adds an extra batch dim
                    else:
                        glyph_inputs_1 = glyph_inputs
                    
                    # Resize the tensor
                    glyph_resize = T.Resize([32, 128])
                    glyph_inputs_resized = torch.stack([glyph_resize(img) for img in glyph_inputs])  # Apply resize per batch
                    
                    # Save each image in the batch
                    for idx, image_tensor in enumerate(glyph_inputs_resized):
                        # Convert tensor to PIL Image
                        image_pil = T.ToPILImage()(image_tensor.cpu())
                        
                        # Save the image
                        image_path = os.path.join(save_dir, f"image_{i+1}_{idx}.png")
                        image_pil.save(image_path)
                    
                    # Resize glyph_inputs for monitoring
                    glyph_inputs = glyph_resize(glyph_inputs)
                    
                    # Monitor and calculate cosine similarity
                    outputs = self.monitor(glyph_inputs)
                    cosine_score = glyph_cosine_similarity(outputs, gt_ids)
                    print("glyph cosine score -=-=-=-=-=-=-=--=-=-=-=-=-=-=->", cosine_score)
                    # Reset controller with cosine score
                    controller.reset_alpha(cosine_score)
                    
                    # Exit the script
                    # import sys
                    # sys.exit()
                # if (i + 1) % 5 == 0:
                #     save_dir = "saved_images"
                #     os.makedirs(save_dir, exist_ok=True)
                #     glyph_inputs = self.latent2image(latents, return_type="pt")
                #     print(glyph_inputs.shape)
                     
                #     if len(glyph_inputs.shape) == 4:  # Already (b, c, h, w)
                #         glyph_inputs_1 = glyph_inputs.unsqueeze(0)  # Adds an extra batch dim

                #     # Resize the tensor
                #     glyph_inputs_1 = glyph_resize(glyph_inputs_1)

                #     # Save each image in the batch
                #     for idx, image_tensor in enumerate(glyph_inputs_1):
                #         # Convert tensor to PIL Image
                #         image_pil = T.ToPILImage()(image_tensor.cpu())
                        
                #         # Save the image
                #         image_path = os.path.join(save_dir, f"image_{i+1}_{idx}.png")
                #         image_pil.save(image_path)
                #         print(f"Saved: {image_path}")
                #     glyph_inputs = glyph_resize(glyph_inputs)
                #     outputs = self.monitor(glyph_inputs) 
                #     cosine_score = glyph_cosine_similarity(outputs, gt_ids) 
                #     img1 = glyp
                #     controller.reset_alpha(cosine_score)
                #     import sys
                #     sys.exit()
        
        if enable_GaMuSA:
            controller.reset_ctrl()
            controller.reset()
        image = self.latent2image(latents, return_type="pt")
        print(image.shape)
        if return_intermediates:
            pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
            return image, latents_list, pred_x0_list
        return image



def glyph_cosine_similarity(output, gt_labels):
    
    pt_logits = nn.Softmax(dim=2)(output['logits'])
    print("pt_logits" , pt_logits.shape)
    print("gt_labels" , gt_labels.shape)
    
    assert pt_logits.shape[0] == gt_labels.shape[0]
    assert pt_logits.shape[2] == gt_labels.shape[2]
    score = nn.CosineSimilarity(dim=2, eps=1e-6)(pt_logits, gt_labels) 
    # print("score -=-=-=-=-==-=-=-=-=-=-=-=-=--> ", score)
    mean_score = torch.mean(score, dim=1)
    print("mean score ", mean_score)
    tensor = torch.tensor([0.5,0.5])
    tensor = tensor.to('cuda')
    # return mean_score
    return tensor 

def prepare_label(labels, charset, max_length, device):
    gt_ids = []
    gt_lengths = []
    for label in labels:
        length = torch.tensor(max_length, dtype=torch.long)
        label = charset.get_labels(label, length=max_length, padding=True, case_sensitive=False)
        label = torch.tensor(label, dtype=torch.long)
        label = CharsetMapper.onehot(label, charset.num_classes)
        gt_ids.append(label)
        gt_lengths.append(length)
    gt_ids = torch.stack(gt_ids).to(device)
    gt_lengths = torch.stack(gt_lengths).to(device)
    
    return gt_ids, gt_lengths