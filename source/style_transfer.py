import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms


IMAGENET_TEMPLATES = (
    'a bad photo of a {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
)


@dataclass
class CLIPGaussianConfig:
    style_prompt: str = ""
    style_image: str = ""
    object_prompt: str = "a Photo"
    clip_model: str = "ViT-B/32"
    clip_backend: str = "auto"
    vgg_weights: str = "default"
    lambda_dir: float = 5.0
    lambda_patch: float = 90.0
    lambda_content: float = 0.8
    lambda_bg: float = 1000.0
    object_background: bool = False
    crop_size: int = 128
    num_crops: int = 64
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class LatentOptimizationConfig:
    steps: int = 5000
    lr: float = 0.005
    save_every: int = 100
    num_views: int = 1
    strength: float = 1.0
    latent_clip: float = 0.25
    latent_mode: str = "shift"
    latent_reg_weight: float = 5e-2
    grad_clip: float = 1.0
    mask_threshold: float = 0.02
    finetune_model: bool = True
    model_lr: float = 5e-3
    model_reg_weight: float = 1e-4
    scale_reg_weight: float = 1e-3
    model_grad_clip: float = 1.0
    freeze_view_branch: bool = True
    finetune_geometry: bool = True
    geometry_lr_m: float = 1e-4
    geometry_lr_s: float = 5e-3
    geometry_lr_q: float = 1e-3
    geometry_lr_A: float = 1e-3
    geometry_lr_k: float = 1e-4
    geometry_grad_clip: float = 1.0
    min_s_coef: float = 0.00034641
    max_s_coef: float = 0.05


APPEARANCE_FINETUNE_KEYS = (
    "w1_uv",
    "w1_v",
    "w1_conditioning",
    "b1",
    "w2",
    "b2",
    "w3",
    "b3",
    "features",
)


GEOMETRY_FINETUNE_KEYS = ("m", "s", "q", "A", "k")


def load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def imagenet_normalize(image: torch.Tensor) -> torch.Tensor:
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (image - mean) / std


def clip_normalize(image: torch.Tensor) -> torch.Tensor:
    image = F.interpolate(image, size=(224, 224), mode="bicubic", align_corners=False)
    mean = image.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = image.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    return (image - mean) / std


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def compose_text_with_templates(text: str, templates: Sequence[str] = IMAGENET_TEMPLATES) -> Sequence[str]:
    return [template.format(text) for template in templates]


class CLIPBackend:
    def __init__(self, model_name: str, backend: str, device: torch.device):
        self.device = device
        self.backend = backend
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self) -> None:
        errors = []
        if self.backend in ("auto", "openai"):
            try:
                import clip

                self.clip = clip
                self.model, _ = clip.load(self.model_name, self.device.type, jit=False)
                self.model.eval()
                self.backend = "openai"
                return
            except Exception as exc:
                errors.append(f"openai clip: {exc}")

        if self.backend in ("auto", "open_clip"):
            try:
                import open_clip

                open_clip_name = self.model_name.replace("/", "-")
                if open_clip_name == "ViT-B-32":
                    # OpenAI ViT-B/32 was trained with QuickGELU.
                    open_clip_name = "ViT-B-32-quickgelu"
                self.model, _, _ = open_clip.create_model_and_transforms(
                    open_clip_name, pretrained="openai", device=self.device
                )
                self.tokenizer = open_clip.get_tokenizer(open_clip_name)
                self.model.eval()
                self.backend = "open_clip"
                return
            except Exception as exc:
                errors.append(f"open_clip: {exc}")

        raise RuntimeError(
            "CLIPGaussian needs either OpenAI CLIP or open_clip for directional CLIP losses. "
            "Install one of them, for example: pip install git+https://github.com/openai/CLIP.git. "
            f"Tried: {'; '.join(errors)}"
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return normalize_features(self.model.encode_image(clip_normalize(image)))

    def encode_text(self, texts: Iterable[str]) -> torch.Tensor:
        if self.backend == "openai":
            tokens = self.clip.tokenize(list(texts)).to(self.device)
        else:
            tokens = self.tokenizer(list(texts)).to(self.device)
        return normalize_features(self.model.encode_text(tokens))


class CLIPGaussianStyleLoss(torch.nn.Module):
    def __init__(self, config: CLIPGaussianConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.needs_clip = config.lambda_dir > 0.0 or config.lambda_patch > 0.0

        self.needs_vgg = config.lambda_content > 0.0
        self.vgg = None
        if self.needs_vgg:
            weights = models.VGG19_Weights.DEFAULT if config.vgg_weights == "default" else None
            self.vgg = models.vgg19(weights=weights).features.eval().to(device)
            for param in self.vgg.parameters():
                param.requires_grad_(False)
        self.vgg_layers = {"21": "conv4_2", "31": "conv5_2"}

        self.clip_backend = None
        self.register_buffer("style_direction", torch.zeros(1, 512, device=device), persistent=False)
        if self.needs_clip:
            self.clip_backend = CLIPBackend(config.clip_model, config.clip_backend, device)
            self.style_direction = self._build_style_direction()

        self.cropper = transforms.RandomCrop(config.crop_size)
        self.augment = transforms.Compose([
            transforms.RandomPerspective(fill=0, p=1.0, distortion_scale=0.5),
            transforms.Resize(224),
        ])

    def _build_style_direction(self) -> torch.Tensor:
        with torch.no_grad():
            if self.config.style_image:
                style_image = load_image_tensor(self.config.style_image, self.device)
                style_features = self.clip_backend.encode_image(style_image).detach()
            elif self.config.style_prompt:
                style_text = compose_text_with_templates(self.config.style_prompt)
                style_features = self.clip_backend.encode_text(style_text).mean(dim=0, keepdim=True).detach()
                style_features = normalize_features(style_features)
            else:
                raise ValueError("CLIPGaussian style transfer needs --style_image or --style_prompt.")

            source_text = compose_text_with_templates(self.config.object_prompt)
            source_features = self.clip_backend.encode_text(source_text).mean(dim=0, keepdim=True).detach()
            source_features = normalize_features(source_features)

            direction = style_features - source_features
            return normalize_features(direction).detach()

    def vgg_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        if self.vgg is None:
            return features
        x = imagenet_normalize(image)
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.vgg_layers:
                features[self.vgg_layers[name]] = x
        return features

    @torch.no_grad()
    def prepare_reference(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        reference = {"vgg": self.vgg_features(image)}
        if self.needs_clip:
            reference["clip"] = self.clip_backend.encode_image(image).detach()
        return reference

    def make_patches(self, image: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        patch_source = image
        h, w = patch_source.shape[-2:]
        crop_size = min(self.config.crop_size, h, w)
        if crop_size < self.config.crop_size:
            scale = self.config.crop_size / max(1, crop_size)
            new_size = (max(self.config.crop_size, int(h * scale)), max(self.config.crop_size, int(w * scale)))
            patch_source = F.interpolate(patch_source, size=new_size, mode="bilinear", align_corners=False)

        patches = []
        for _ in range(max(1, self.config.num_crops)):
            patch = self.cropper(patch_source)
            patch = self.augment(patch)
            patches.append(patch)
        return torch.cat(patches, dim=0)

    def background_loss(self, image: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.config.object_background or mask is None:
            return image.new_zeros(())
        background = image.new_tensor(self.config.background).view(1, 3, 1, 1)
        bg_mask = 1.0 - mask
        denom = bg_mask.sum().clamp_min(1.0)
        return ((image - background).abs() * bg_mask).sum() / denom

    def forward(
        self,
        image: torch.Tensor,
        reference_image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reference: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        reference_image = torch.nan_to_num(reference_image, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if reference is None:
            reference = self.prepare_reference(reference_image)

        render_features = self.vgg_features(image)
        content = image.new_zeros(())
        for key, target in reference["vgg"].items():
            content = content + F.mse_loss(render_features[key], target.detach())

        background = self.background_loss(image, mask)
        directional = image.new_zeros(())
        patch = image.new_zeros(())

        if self.needs_clip:
            source_clip = reference["clip"].detach()
            render_clip = self.clip_backend.encode_image(image)
            image_direction = normalize_features(render_clip - source_clip)
            directional = (1.0 - F.cosine_similarity(image_direction, self.style_direction, dim=-1)).mean()

            patches = self.make_patches(image, mask)
            patch_features = self.clip_backend.encode_image(patches)
            patch_direction = normalize_features(patch_features - source_clip.repeat(patch_features.shape[0], 1))
            patch = (1.0 - F.cosine_similarity(patch_direction, self.style_direction.repeat(patch_features.shape[0], 1), dim=-1)).mean()

        total = (
            self.config.lambda_dir * directional
            + self.config.lambda_patch * patch
            + self.config.lambda_content * content
            + self.config.lambda_bg * background
        )
        return total, {
            "total": total.detach(),
            "directional": directional.detach(),
            "patch": patch.detach(),
            "content": content.detach(),
            "background": background.detach(),
        }


def crop_to_mask(image: torch.Tensor, mask: Optional[torch.Tensor], padding_fraction: float = 0.08) -> torch.Tensor:
    if mask is None:
        return image
    coords = torch.nonzero(mask[0, 0] > 0.5, as_tuple=False)
    if coords.numel() == 0:
        return image

    y0 = int(coords[:, 0].min().item())
    y1 = int(coords[:, 0].max().item()) + 1
    x0 = int(coords[:, 1].min().item())
    x1 = int(coords[:, 1].max().item()) + 1
    pad_y = max(2, int((y1 - y0) * padding_fraction))
    pad_x = max(2, int((x1 - x0) * padding_fraction))
    y0 = max(0, y0 - pad_y)
    y1 = min(image.shape[2], y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(image.shape[3], x1 + pad_x)
    return image[:, :, y0:y1, x0:x1]


def object_mask_from_render(image: torch.Tensor, background: Tuple[float, float, float], threshold: float) -> torch.Tensor:
    bg = image.new_tensor(background).view(1, 3, 1, 1)
    return ((image - bg).abs().amax(dim=1, keepdim=True) > threshold).float()


def flat_rgb_to_bchw(flat_rgb: torch.Tensor, height: int, width: int) -> torch.Tensor:
    flat_rgb = torch.nan_to_num(flat_rgb, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.clamp(flat_rgb, 0.0, 1.0).reshape(height, width, 3).permute(2, 0, 1).unsqueeze(0)


def camera_original_to_bchw(camera, height: int, width: int, device: torch.device) -> Optional[torch.Tensor]:
    original = getattr(camera, "original_image", None)
    if original is None:
        return None
    original = torch.nan_to_num(original[:3].to(device), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if original.dim() != 3:
        return None
    image = original.unsqueeze(0)
    if image.shape[-2:] != (height, width):
        image = F.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)
    return image.detach()


def save_flat_rgb(flat_rgb: torch.Tensor, height: int, width: int, path: str) -> None:
    image = torch.nan_to_num(flat_rgb, nan=0.0, posinf=1.0, neginf=0.0)
    image = torch.clamp(image, 0.0, 1.0).reshape(height, width, 3)
    arr = (image.detach().cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def latent_style_conditioning(
    base: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    strength: float,
    latent_clip: float,
    mode: str,
) -> torch.Tensor:
    mu = base.mean(dim=0, keepdim=True)
    sigma = base.std(dim=0, keepdim=True).clamp_min(1e-5)
    normalized = (base - mu) / sigma

    if latent_clip > 0.0:
        gamma_delta = torch.tanh(gamma).view(1, -1) * latent_clip
        beta_delta = torch.tanh(beta).view(1, -1) * sigma * latent_clip
    else:
        gamma_delta = gamma.view(1, -1)
        beta_delta = beta.view(1, -1) * sigma

    if mode == "shift":
        residual = beta_delta
    elif mode == "affine":
        residual = normalized * gamma_delta + beta_delta
    else:
        raise ValueError(f"Unknown latent mode {mode!r}. Expected 'shift' or 'affine'.")
    return base + strength * residual


def prepare_style_checkpoint(
    checkpoint: Dict[str, torch.Tensor],
    finetune_keys: Sequence[str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    style_checkpoint = dict(checkpoint)
    originals = {}
    for key in finetune_keys:
        if key not in style_checkpoint:
            continue
        originals[key] = style_checkpoint[key].detach().clone()
        style_checkpoint[key] = style_checkpoint[key].detach().clone().requires_grad_(True)
    return style_checkpoint, originals


def style_optimizer_param_groups(
    params: Dict[str, torch.Tensor],
    latent_config: LatentOptimizationConfig,
) -> Sequence[Dict[str, object]]:
    groups = []
    appearance = [param for key, param in params.items() if key in APPEARANCE_FINETUNE_KEYS]
    if appearance:
        groups.append({"params": appearance, "lr": latent_config.model_lr})

    geometry_lrs = {
        "m": latent_config.geometry_lr_m,
        "s": latent_config.geometry_lr_s,
        "q": latent_config.geometry_lr_q,
        "A": latent_config.geometry_lr_A,
        "k": latent_config.geometry_lr_k,
    }
    for key, lr in geometry_lrs.items():
        param = params.get(key)
        if param is not None and lr > 0.0:
            groups.append({"params": [param], "lr": lr})
    return groups


def clamp_geometry_params(
    checkpoint: Dict[str, torch.Tensor],
    min_s_coef: float,
    max_s_coef: float,
) -> None:
    if "s" not in checkpoint or "m" not in checkpoint:
        return
    with torch.no_grad():
        extent = torch.sqrt(((checkpoint["m"].max(0, keepdim=True)[0] - checkpoint["m"].min(0, keepdim=True)[0]) ** 2).sum(1)).item()
        max_s = max(min_s_coef, max_s_coef * max(extent / 2.0, min_s_coef))
        checkpoint["s"].data.clamp_(min=float(np.log(min_s_coef)), max=float(np.log(max_s)))
        for key in GEOMETRY_FINETUNE_KEYS:
            if key in checkpoint:
                checkpoint[key].data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)


def assign_model_grads(
    params: Dict[str, torch.Tensor],
    grads: Dict[str, torch.Tensor],
    scale: float,
) -> None:
    for key, param in params.items():
        grad = grads.get(key)
        if grad is None:
            continue
        grad = grad.detach() * scale
        param.grad = grad if param.grad is None else param.grad + grad


def model_delta_norm(params: Dict[str, torch.Tensor], originals: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key, param in params.items():
        total += float((param.detach() - originals[key]).norm())
    return total


def sample_view_indices(num_cameras: int, batch_size: int, device: torch.device) -> Sequence[int]:
    batch_size = max(1, min(batch_size, num_cameras))
    if batch_size == num_cameras:
        return list(range(num_cameras))
    return torch.randperm(num_cameras, device=device)[:batch_size].cpu().tolist()


def tensor_tree_to(value, device, detach: bool = False):
    """Move tensors in nested reference dictionaries between CPU and CUDA."""
    if isinstance(value, dict):
        return {key: tensor_tree_to(item, device, detach) for key, item in value.items()}
    if torch.is_tensor(value):
        value = value.detach() if detach else value
        return value.to(device)
    return value


def model_regularization(
    params: Dict[str, torch.Tensor],
    originals: Dict[str, torch.Tensor],
    weight: float,
) -> torch.Tensor:
    if weight <= 0.0 or not params:
        first = next(iter(params.values()), None)
        if first is None:
            return torch.zeros((), dtype=torch.float32)
        return first.new_zeros(())

    reg = None
    for key, param in params.items():
        term = F.mse_loss(param, originals[key])
        reg = term if reg is None else reg + term
    return weight * reg


def apply_scale_regularization(
    geometry_params: Dict[str, torch.Tensor],
    weight: float,
) -> None:
    s = geometry_params.get("s")
    if s is None or s.grad is None or weight <= 0.0:
        return
    with torch.no_grad():
        s_squared = torch.exp(s.detach()) ** 2
        denom = torch.sqrt(s_squared.sum(1, keepdim=True)).clamp_min(1e-12)
        s.grad.add_((weight / max(1, s.shape[0])) * (s_squared / denom))


def tensor_stats(prefix: str, tensor: torch.Tensor) -> str:
    value = tensor.detach()
    finite = torch.isfinite(value)
    if not bool(finite.any()):
        return f"{prefix}_mean=nan {prefix}_min=nan {prefix}_max=nan"
    value = value[finite]
    return f"{prefix}_mean={float(value.mean())} {prefix}_min={float(value.min())} {prefix}_max={float(value.max())}"


def geometry_log_stats(geometry_params: Dict[str, torch.Tensor]) -> str:
    if not geometry_params:
        return ""
    fields = []
    for key in ("m", "s", "q", "A", "k"):
        param = geometry_params.get(key)
        if param is None:
            continue
        if param.grad is not None:
            fields.append(f"{key}_grad_norm={float(param.grad.detach().norm())}")
        fields.append(tensor_stats(key, param))
        if key == "A":
            fields.append(tensor_stats("opacity", torch.sigmoid(param)))
        elif key == "s":
            fields.append(tensor_stats("scale", torch.exp(param)))
        elif key == "k":
            fields.append(tensor_stats("sharpness", 1.0 + F.softplus(param)))
    return " ".join(fields)


def optimize_latent_style(
    *,
    style_config: CLIPGaussianConfig,
    latent_config: LatentOptimizationConfig,
    renderer,
    checkpoint: Dict[str, torch.Tensor],
    base_conditioning: torch.Tensor,
    cameras: Sequence,
    width: int,
    height: int,
    render_dir: str,
    forward_inference: Callable,
    forward_training: Callable,
    backward_conditioning: Callable,
    render_args,
    backward_model: Optional[Callable] = None,
    update_geometry: Optional[Callable] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    style_dir = os.path.join(render_dir, "style_optimization")
    os.makedirs(style_dir, exist_ok=True)

    device = base_conditioning.device
    loss_fn = CLIPGaussianStyleLoss(style_config, device).to(device)
    all_cameras = list(cameras)
    views_per_step = max(1, min(latent_config.num_views, len(all_cameras)))
    reference_checkpoint = dict(checkpoint)
    base_conditioning = base_conditioning.detach()
    model_params = {}
    model_originals = {}
    geometry_params = {}
    model_optimizer = None
    if latent_config.finetune_model and backward_model is not None:
        finetune_keys = list(APPEARANCE_FINETUNE_KEYS)
        if latent_config.freeze_view_branch and "w1_v" in finetune_keys:
            finetune_keys.remove("w1_v")
        if latent_config.finetune_geometry:
            finetune_keys.extend(GEOMETRY_FINETUNE_KEYS)
        checkpoint, model_originals = prepare_style_checkpoint(checkpoint, finetune_keys)
        model_params = {key: checkpoint[key] for key in model_originals}
        geometry_params = {key: checkpoint[key] for key in GEOMETRY_FINETUNE_KEYS if key in model_params}
        param_groups = style_optimizer_param_groups(model_params, latent_config)
        if param_groups:
            model_optimizer = torch.optim.Adam(param_groups)

    gamma = torch.zeros(base_conditioning.shape[1], dtype=torch.float32, device=device, requires_grad=True)
    beta = torch.zeros(base_conditioning.shape[1], dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([gamma, beta], lr=latent_config.lr)

    background = style_config.background

    # Preserve what the loaded FlaRe model actually renders. Dataset photos can
    # differ from the reconstruction and make the content/source losses fight
    # both reconstruction error and stylization at the same time. Precompute all
    # cameras before geometry optimization so every reference uses one immutable
    # baseline state. Keep the cache on CPU to avoid retaining VGG/CLIP features
    # for every camera in GPU memory.
    baseline_reference_cache = {}
    with torch.no_grad():
        for camera_idx, camera in enumerate(
            tqdm(all_cameras, desc="Caching baseline references", dynamic_ncols=True)
        ):
            baseline_flat = forward_inference(
                renderer,
                reference_checkpoint,
                camera,
                base_conditioning,
                render_args,
                width,
                height,
            )
            baseline_image = flat_rgb_to_bchw(baseline_flat, height, width).detach()
            baseline_mask = object_mask_from_render(
                baseline_image, background, latent_config.mask_threshold
            ).detach()
            baseline_features = loss_fn.prepare_reference(baseline_image)
            baseline_reference_cache[camera_idx] = (
                baseline_image.cpu(),
                baseline_mask.cpu(),
                tensor_tree_to(baseline_features, "cpu", detach=True),
            )

    def get_view_reference(camera_idx: int):
        original, mask, reference = baseline_reference_cache[camera_idx]
        return (
            original.to(device),
            mask.to(device),
            tensor_tree_to(reference, device),
        )

    log_path = os.path.join(style_dir, "losses.txt")
    with open(log_path, "w", encoding="utf-8") as log:
        progress = tqdm(range(latent_config.steps), desc="Style optimization", dynamic_ncols=True)
        for step in progress:
            optimizer.zero_grad(set_to_none=True)
            if model_optimizer is not None:
                model_optimizer.zero_grad(set_to_none=True)
            styled_conditioning = latent_style_conditioning(
                base_conditioning,
                gamma,
                beta,
                latent_config.strength,
                latent_config.latent_clip,
                latent_config.latent_mode,
            )
            styled_fp16 = styled_conditioning.detach().to(torch.float16)
            grad_conditioning = torch.zeros_like(base_conditioning)
            metric_sums = {"total": 0.0, "directional": 0.0, "patch": 0.0, "content": 0.0, "background": 0.0}
            batch_indices = sample_view_indices(len(all_cameras), views_per_step, device)

            for camera_idx in batch_indices:
                camera = all_cameras[camera_idx]
                original, mask, reference = get_view_reference(camera_idx)
                img_unclamped, *camera_cache = forward_training(
                    renderer, checkpoint, camera, styled_fp16, render_args, width, height
                )
                finite = torch.isfinite(img_unclamped)
                img_clean = torch.nan_to_num(img_unclamped, nan=0.0, posinf=1.0, neginf=0.0)
                img_for_loss = torch.clamp(img_clean, 0.0, 1.0).detach().requires_grad_(True)
                image = flat_rgb_to_bchw(img_for_loss, height, width)

                loss, metrics = loss_fn(image, original, mask, reference)
                loss.backward()

                dloss_dimage = img_for_loss.grad.detach() * finite
                if model_optimizer is None:
                    grad_conditioning += backward_conditioning(
                        renderer,
                        checkpoint,
                        styled_fp16,
                        tuple(camera_cache),
                        img_unclamped,
                        img_for_loss.detach(),
                        dloss_dimage,
                        render_args,
                        width,
                        height,
                    )
                else:
                    model_grads = backward_model(
                        renderer,
                        checkpoint,
                        styled_fp16,
                        tuple(camera_cache),
                        img_unclamped,
                        img_for_loss.detach(),
                        dloss_dimage,
                        render_args,
                        width,
                        height,
                    )
                    grad_conditioning += model_grads["conditioning_variable"]
                    assign_model_grads(model_params, model_grads, 1.0 / len(batch_indices))
                for key in metric_sums:
                    metric_sums[key] += float(metrics[key])

            latent_reg = latent_config.latent_reg_weight * F.mse_loss(styled_conditioning, base_conditioning)
            latent_reg.backward(retain_graph=True)
            styled_conditioning.backward(grad_conditioning / len(batch_indices))
            torch.nn.utils.clip_grad_norm_([gamma, beta], latent_config.grad_clip)
            model_reg = model_regularization(model_params, model_originals, latent_config.model_reg_weight)
            if model_optimizer is not None:
                model_reg.backward()
                appearance_params = [param for key, param in model_params.items() if key in APPEARANCE_FINETUNE_KEYS]
                if appearance_params:
                    torch.nn.utils.clip_grad_norm_(appearance_params, latent_config.model_grad_clip)
                if geometry_params:
                    apply_scale_regularization(geometry_params, latent_config.scale_reg_weight)
                    torch.nn.utils.clip_grad_norm_(geometry_params.values(), latent_config.geometry_grad_clip)
                model_optimizer.step()
                if geometry_params:
                    clamp_geometry_params(checkpoint, latent_config.min_s_coef, latent_config.max_s_coef)
                    if update_geometry is not None:
                        update_geometry(renderer, checkpoint)
            optimizer.step()

            with torch.no_grad():
                preview_conditioning = latent_style_conditioning(
                    base_conditioning,
                    gamma.detach(),
                    beta.detach(),
                    latent_config.strength,
                    latent_config.latent_clip,
                    latent_config.latent_mode,
                )
                preview_flat = forward_inference(
                    renderer, checkpoint, all_cameras[batch_indices[0]], preview_conditioning, render_args, width, height
                )
                preview_stats = torch.nan_to_num(preview_flat, nan=0.0, posinf=1.0, neginf=0.0)

            if step % latent_config.save_every == 0 or step == latent_config.steps - 1:
                save_flat_rgb(preview_flat, height, width, os.path.join(style_dir, f"step_{step:05d}.png"))

            n = len(batch_indices)
            objective = metric_sums["total"] / n + float(latent_reg.detach()) + float(model_reg.detach())
            progress.set_postfix(
                total=f"{objective:.4f}",
                direction=f"{metric_sums['directional'] / n:.4f}",
                patch=f"{metric_sums['patch'] / n:.4f}",
                content=f"{metric_sums['content'] / n:.4f}",
            )
            log.write(
                f"{step} objective={objective} "
                f"directional={metric_sums['directional'] / n} patch={metric_sums['patch'] / n} "
                f"content={metric_sums['content'] / n} background={metric_sums['background'] / n} "
                f"latent={float(latent_reg.detach())} model={float(model_reg.detach())} "
                f"image_min={float(preview_stats.min())} "
                f"image_mean={float(preview_stats.mean())} image_max={float(preview_stats.max())} "
                f"gamma_norm={float(gamma.detach().norm())} beta_norm={float(beta.detach().norm())} "
                f"model_delta_norm={model_delta_norm(model_params, model_originals)} "
                f"geometry_delta_norm={model_delta_norm(geometry_params, model_originals)} "
                f"{geometry_log_stats(geometry_params)} "
                f"views={','.join(str(idx) for idx in batch_indices)}\n"
            )
            log.flush()

    styled_conditioning = latent_style_conditioning(
        base_conditioning,
        gamma.detach(),
        beta.detach(),
        latent_config.strength,
        latent_config.latent_clip,
        latent_config.latent_mode,
    ).detach()
    torch.save(
        {
            "gamma": gamma.detach(),
            "beta": beta.detach(),
            "conditioning_variable": styled_conditioning,
            "model_state": {key: value.detach() for key, value in model_params.items()},
            "clipgaussian_config": style_config,
            "latent_config": latent_config,
        },
        os.path.join(style_dir, "latent_style.pt"),
    )
    checkpoint["conditioning_variable"] = styled_conditioning
    tensor_names = (
        "RGB", "A", "k", "w1_uv", "w1_v", "w1_conditioning", "b1",
        "w2", "b2", "w3", "b3", "conditioning_variable", "features",
        "m", "s", "q",
    )
    if "RGB" not in checkpoint:
        tensor_names = tensor_names[1:]
    payload = tuple(checkpoint[name].detach() for name in tensor_names) + (
        None, float(checkpoint.get("training_time_seconds", 0.0)),
    )
    torch.save(payload, os.path.join(style_dir, "styled_model.checkpoint"))
    return styled_conditioning, checkpoint
