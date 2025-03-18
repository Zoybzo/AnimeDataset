from trainers import Trainer


class VaeTester(Trainer):
    def __init__(self, model_path, subfolder="vae", device="cpu"):
        self.model_path = model_path
        self.subfolder = subfolder
        self.device = device
        self.prepare_models(self.model_path, self.subfolder, self.device)
        assert self.vae is not None

    def prepare_models(self, model_path=None, subfolder=None, device="cpu"):
        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            subfolder=self.subfolder,
        ).to(self.device)
        self.vae.enable_slicing()
        self.vae.enable_tiling()

    def validate(self, dataloader):
        for features in next(iter(dataloader)):
            print(features.size())
