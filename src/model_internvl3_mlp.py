import torch
import torch.nn as nn
import torch.nn.functional as F


class InternVL3MLPPolicy(nn.Module):
    """InternVL3 visual-language backbone with a small MLP action head.

    This follows the Being-H0 interface idea: the prompt contains an image,
    a proprio placeholder, and an action placeholder. For robustness with the
    stock InternVL3 Hugging Face model, proprioception is fused after the VLM
    forward pass instead of patching InternVL internals.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        model_name="OpenGVLab/InternVL3-2B",
        task_description="",
        freeze_vlm=True,
        tune_mode=None,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=None,
        torch_dtype="bfloat16",
        image_size=None,
        trust_remote_code=True,
        local_files_only=False,
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for InternVL3MLPPolicy. "
                "Install it in the training environment."
            ) from exc

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.model_name = model_name
        self.task_description = task_description
        self.tune_mode = tune_mode or ("frozen" if freeze_vlm else "full")
        if self.tune_mode not in {"frozen", "lora", "full"}:
            raise ValueError("tune_mode must be one of: frozen, lora, full")
        self.freeze_vlm = self.tune_mode == "frozen"
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        self.vlm_dtype = self._resolve_dtype(torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        self.vlm = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.vlm_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )

        self._ensure_special_tokens()

        self.hidden_size = self._get_hidden_size()
        self.image_size = image_size or self._get_image_size(default=448)
        self.num_image_token = self._get_num_image_token()
        self._configure_vlm_tuning()

        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.proprio_encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size),
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 2),
            nn.Linear(self.hidden_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )

    def _configure_vlm_tuning(self):
        if self.tune_mode == "frozen":
            self.vlm.eval()
            for param in self.vlm.parameters():
                param.requires_grad_(False)
            return

        if self.tune_mode == "full":
            for param in self.vlm.parameters():
                param.requires_grad_(True)
            language_model = getattr(self.vlm, "language_model", None)
            lm_head = getattr(language_model, "lm_head", None)
            if lm_head is not None:
                for param in lm_head.parameters():
                    param.requires_grad_(False)
            return

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "peft is required for --tune_mode lora. Install peft or use "
                "--tune_mode frozen/full."
            ) from exc

        for param in self.vlm.parameters():
            param.requires_grad_(False)

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.vlm = get_peft_model(self.vlm, lora_config)
        if hasattr(self.vlm, "print_trainable_parameters"):
            self.vlm.print_trainable_parameters()

    @staticmethod
    def _resolve_dtype(torch_dtype):
        if torch_dtype in ("bf16", "bfloat16"):
            return torch.bfloat16
        if torch_dtype in ("fp16", "float16", "half"):
            return torch.float16
        if torch_dtype in ("fp32", "float32", "float"):
            return torch.float32
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

    def _ensure_special_tokens(self):
        if hasattr(self.vlm, "img_context_token_id"):
            self.vlm.img_context_token_id = self.tokenizer.convert_tokens_to_ids(
                "<IMG_CONTEXT>"
            )

        # Do not add new tokens to the stock InternVL3 vocabulary. The remote
        # InternVLChatModel does not expose the full embedding resize API.
        self.prop_token_id = None
        self.action_token_id = None

    def _get_hidden_size(self):
        language_model = getattr(self.vlm, "language_model", None)
        if language_model is not None and hasattr(language_model, "config"):
            hidden_size = getattr(language_model.config, "hidden_size", None)
            if hidden_size is not None:
                return hidden_size

        config = getattr(self.vlm, "config", None)
        llm_config = getattr(config, "llm_config", None)
        if llm_config is not None:
            hidden_size = getattr(llm_config, "hidden_size", None)
            if hidden_size is not None:
                return hidden_size
            if isinstance(llm_config, dict) and "hidden_size" in llm_config:
                return llm_config["hidden_size"]

        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer InternVL hidden size from the model config.")
        return hidden_size

    def _get_image_size(self, default=448):
        config = getattr(self.vlm, "config", None)
        force_image_size = getattr(config, "force_image_size", None)
        if force_image_size:
            return force_image_size

        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            image_size = getattr(vision_config, "image_size", None)
            if image_size is not None:
                return image_size
            if isinstance(vision_config, dict) and "image_size" in vision_config:
                return vision_config["image_size"]

        return default

    def _get_num_image_token(self):
        num_image_token = getattr(self.vlm, "num_image_token", None)
        if num_image_token is not None:
            return int(num_image_token)

        config = getattr(self.vlm, "config", None)
        downsample_ratio = getattr(config, "downsample_ratio", 0.5)
        vision_config = getattr(config, "vision_config", None)
        patch_size = getattr(vision_config, "patch_size", 14)
        return int((self.image_size // patch_size) ** 2 * (downsample_ratio**2))

    def _preprocess_images(self, imgs):
        # Dataset stores three RGB frames as [B, 9, 224, 224]. Use the latest frame
        # to match standard VLM single-image inputs.
        if imgs.dim() != 4:
            raise ValueError(f"Expected image tensor [B, C, H, W], got {tuple(imgs.shape)}")
        if imgs.size(1) == 9:
            imgs = imgs[:, -3:]
        elif imgs.size(1) != 3:
            raise ValueError(f"Expected 3 or 9 image channels, got {imgs.size(1)}")

        imgs = imgs.float().clamp(0.0, 1.0)
        imgs = F.interpolate(
            imgs,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )
        imgs = (imgs - self.image_mean.to(imgs.device)) / self.image_std.to(imgs.device)
        return imgs.to(dtype=self.vlm_dtype)

    def _make_prompts(self, batch_size):
        image_tokens = "<img>" + ("<IMG_CONTEXT>" * self.num_image_token) + "</img>"
        task = self.task_description or "What is the next action for the robot?"
        prompt = (
            f"{image_tokens}\n"
            f"According to the instruction '{task}', predict the next robot action. "
            f"<PROP_CONTEXT> <ACTION_CHUNK_0>"
        )
        return [prompt] * batch_size

    def _extract_action_hidden(self, pixel_values):
        prompts = self._make_prompts(pixel_values.size(0))
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(pixel_values.device)
        attention_mask = inputs["attention_mask"].to(pixel_values.device)
        image_flags = torch.ones(pixel_values.size(0), dtype=torch.long, device=pixel_values.device)

        input_embeds = self.vlm.language_model.get_input_embeddings()(input_ids).clone()
        vit_embeds = self.vlm.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]

        batch_size, seq_len, hidden_size = input_embeds.shape
        flat_embeds = input_embeds.reshape(batch_size * seq_len, hidden_size)
        flat_input_ids = input_ids.reshape(batch_size * seq_len)
        selected = flat_input_ids == self.vlm.img_context_token_id
        flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_embeds.reshape(-1, hidden_size)
        input_embeds = flat_embeds.reshape(batch_size, seq_len, hidden_size)

        language_backbone = getattr(self.vlm.language_model, "model", self.vlm.language_model)
        outputs = language_backbone(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state

        last_token_idx = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, last_token_idx].view(pixel_values.size(0), self.hidden_size).float()

    def train(self, mode=True):
        super().train(mode)
        if self.tune_mode == "frozen":
            self.vlm.eval()
        return self

    def forward(self, img, state):
        pixel_values = self._preprocess_images(img)
        if self.tune_mode == "frozen":
            with torch.no_grad():
                action_hidden = self._extract_action_hidden(pixel_values)
        else:
            action_hidden = self._extract_action_hidden(pixel_values)

        proprio_hidden = self.proprio_encoder(state.float())
        fused = torch.cat([action_hidden, proprio_hidden], dim=-1)
        return self.action_head(fused)

    def trainable_state_dict(self):
        trainable_params = {
            name: param.detach().cpu()
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        return {
            "trainable_params": trainable_params,
            "config": {
                "model_name": self.model_name,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "task_description": self.task_description,
                "image_size": self.image_size,
                "tune_mode": self.tune_mode,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "lora_target_modules": self.lora_target_modules,
            },
        }
