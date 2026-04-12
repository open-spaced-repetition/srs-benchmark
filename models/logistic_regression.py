import torch
from torch import nn, Tensor

from config import Config
from models.base import BaseModel

class LogisticRegression(BaseModel):
    n_epoch: int = 8
    batch_size: int = 1024
    lr: float = 1e-1
    betas: tuple = (0.8, 0.85)
    wd: float = 1e-2

    def __init__(
        self, config: Config, state_dict=None
    ):
        super().__init__(config)
        mean = torch.tensor([-0.9125, -0.7233, -0.5761, -0.4087,  0.2976,  0.0926,  0.3503, -0.3414,
            -0.3245, -0.2032,  1.5807,  0.0816,  0.3976,  0.5160,  0.2769,  0.5066,
            0.6300,  1.3532,  1.3429,  1.4083, -0.7107,  0.4998,  0.1423, -0.2108,
            0.2767,  0.2703,  0.4023,  0.4328, -0.2254])
        self.register_buffer('mean', mean)
        std = torch.tensor([0.2519, 0.2501, 0.2121, 0.1172, 0.1461, 0.1370, 0.1457, 0.1070, 0.0819,
            0.0736, 0.4088, 0.2184, 0.2105, 0.1901, 0.1622, 0.1910, 0.1691, 0.3706,
            0.6443, 0.3314, 0.1577, 0.0584, 0.1096, 0.2270, 0.0770, 0.0682, 0.0876,
            0.1757, 0.2187])
        self.register_buffer('std', std)
        self.n_features = mean.size(0)
        self.coef_res = nn.Parameter(torch.zeros(self.n_features))
        if state_dict is not None:
            self.load_state_dict(state_dict)

    @property
    def coefficients(self):
        return self.coef_res * self.std + self.mean

    def batch_process(
        self,
        sequences: Tensor,
        delta_n: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        feature_elapsed_days_real_bl, feature_elapsed_days_int_bl, label_elapsed_days_real_bl, label_elapsed_days_int_bl, feature_rating_bl = sequences.transpose(0, 1).unbind(dim=-1)
        B, L = feature_rating_bl.shape
        same_day = (feature_elapsed_days_real_bl < 1).float()
        feature_rating_is_hard = (feature_rating_bl == 2).float()
        feature_rating_is_better_than_hard = (feature_rating_bl > 2).float()
        feature_rating_onehot_bl3 = (feature_rating_bl > 1).float().unsqueeze(-1) * torch.nn.functional.one_hot((feature_rating_bl.long() - 2).clamp(min=0), num_classes=3).float()
        success_bl = (feature_rating_bl > 1).float()
        is_first_review_bl = torch.zeros_like(feature_rating_bl)
        is_first_review_bl[:, 0] = 1
        first_rating_bl = feature_rating_bl[:, 0].unsqueeze(-1).expand(B, L)
        first_rating_onehot_bl3 = (first_rating_bl > 1).float().unsqueeze(-1) * torch.nn.functional.one_hot((first_rating_bl.long() - 2).clamp(min=0), num_classes=3).float()

        same_day_fail = same_day * (1 - is_first_review_bl) * (feature_rating_bl == 1.0).float()
        non_same_day_fail = (1 - same_day) * (1 - is_first_review_bl) * (feature_rating_bl == 1.0).float()
        num_same_day_fail = torch.cumsum(same_day_fail, dim=-1)
        num_non_same_day_fail = torch.cumsum(non_same_day_fail, dim=-1)
        same_day_pass = same_day * (1 - is_first_review_bl) * (feature_rating_bl > 1).float()
        non_same_day_pass = (1 - same_day) * (1 - is_first_review_bl) * (feature_rating_bl > 1).float()
        num_same_day_pass = torch.cumsum(same_day_pass, dim=-1)
        num_non_same_day_pass = torch.cumsum(non_same_day_pass, dim=-1)
        num_pass = torch.cumsum((feature_rating_bl > 1).float(), dim=-1)
        has_passed = (num_pass > 0).float()

        times = feature_elapsed_days_real_bl.cumsum(dim=-1)
        first_or_lapse_time = torch.where((is_first_review_bl.bool() | (1 - success_bl).bool()), times, torch.zeros_like(times))
        time_since_first_or_lapse = times - torch.cummax(first_or_lapse_time, dim=-1).values
        
        label_is_same_day = (label_elapsed_days_int_bl == 0).float()
        deg_1_in_blh = torch.concat(
            (
                torch.ones_like(feature_rating_bl).unsqueeze(-1),
                feature_rating_onehot_bl3,
                (feature_rating_is_hard * is_first_review_bl).unsqueeze(-1),
                (feature_rating_is_better_than_hard * is_first_review_bl).unsqueeze(-1),
                (1 + num_same_day_fail).log().unsqueeze(-1),
                (1 + num_non_same_day_pass * (1 - label_is_same_day)).log().unsqueeze(-1),
                (1 + num_non_same_day_fail * (1 - label_is_same_day)).log().unsqueeze(-1),
                (1 + num_same_day_pass).log().unsqueeze(-1),
            ),
            dim=-1
        )
        deg_0_in_blh = torch.concat(
            (
                torch.ones_like(feature_rating_bl).unsqueeze(-1),
                feature_rating_onehot_bl3,
                first_rating_onehot_bl3,
                first_rating_onehot_bl3 * is_first_review_bl.unsqueeze(-1),  
                (1 + num_same_day_fail).log().unsqueeze(-1),
                (1 + num_non_same_day_pass * (1 - label_is_same_day)).log().unsqueeze(-1),
                transform_elapsed_days_real(feature_elapsed_days_real_bl).unsqueeze(-1),
                (1 + num_non_same_day_fail).log().unsqueeze(-1),
                (1 + num_non_same_day_pass).log().unsqueeze(-1),
                (1 + num_same_day_pass * label_is_same_day).log().unsqueeze(-1),
                has_passed.unsqueeze(-1),
                transform_elapsed_days_real(time_since_first_or_lapse).unsqueeze(-1),
                label_is_same_day.unsqueeze(-1),
            ),
            dim=-1
        )
        v_bl = transform_elapsed_days_real(label_elapsed_days_real_bl)
        y_blH = torch.cat(
            (
                deg_1_in_blh * v_bl.unsqueeze(-1),
                deg_0_in_blh,
            )
            , dim=-1
        )
        logit_bl = torch.einsum('blh,h->bl', y_blH, self.coefficients)
        p_bl = 1e-5 + (1 - 2e-5) * torch.sigmoid(logit_bl)
        p_b = torch.gather(p_bl, dim=1, index=(seq_lens - 1).long().clamp_min(0).unsqueeze(-1)).squeeze(-1)
        return {
            "retentions": p_b,
        }

    def get_optimizer(
        self, lr: float, wd: float, betas
    ) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd, betas=betas)

    def log(self, x):
        params = [round(x, 4) for x in self.coefficients.tolist()]
        x["parameters"] = params
        x["state"] = [round(x, 4) for x in self.coef_res.tolist()]

def transform_elapsed_days_real(x):
    return ((x + 1e-5).log() + 1.3) / 5
    