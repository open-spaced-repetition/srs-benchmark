import torch
from torch import nn, Tensor
import numpy as np
from scipy.optimize import minimize  # type: ignore
from tqdm.auto import tqdm  # type: ignore
from config import Config


class FSRS(nn.Module):
    def __init__(self, config: Config):
        super(FSRS, self).__init__()
        self.config = config

    def forgetting_curve(self, t, s):
        raise NotImplementedError("Forgetting curve not implemented")

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs, _ = self.forward(sequences)
        stabilities, difficulties, *_ = outputs[
            seq_lens - 1,
            torch.arange(real_batch_size, device=self.config.device),
        ].transpose(0, 1)
        retentions = self.forgetting_curve(delta_ts, stabilities)
        return {
            "retentions": retentions,
            "stabilities": stabilities,
            "difficulties": difficulties,
        }

    def pretrain(self, train_set):
        S0_dataset_group = (
            train_set[train_set["i"] == 2]
            .groupby(by=["first_rating", "delta_t"], group_keys=False)
            .agg({"y": ["mean", "count"]})
            .reset_index()
        )
        rating_stability = {}
        rating_count = {}
        average_recall = train_set["y"].mean()
        r_s0_default = {str(i): self.init_w[i - 1] for i in range(1, 5)}

        for first_rating in ("1", "2", "3", "4"):
            group = S0_dataset_group[S0_dataset_group["first_rating"] == first_rating]
            if group.empty:
                if self.config.verbose_inadequate_data:
                    tqdm.write(
                        f"Not enough data for first rating {first_rating}. Expected at least 1, got 0."
                    )
                continue
            delta_t = group["delta_t"]
            if self.config.use_secs_intervals:
                recall = group["y"]["mean"]
            else:
                recall = (
                    group["y"]["mean"] * group["y"]["count"] + average_recall * 1
                ) / (group["y"]["count"] + 1)
            count = group["y"]["count"]

            init_s0 = r_s0_default[first_rating]

            def loss(stability):
                y_pred = self.forgetting_curve(delta_t, stability)
                logloss = sum(
                    -(recall * np.log(y_pred) + (1 - recall) * np.log(1 - y_pred))
                    * count
                )
                l1 = (
                    np.abs(stability - init_s0) / 16
                    if not self.config.use_secs_intervals
                    else 0
                )
                return logloss + l1

            res = minimize(
                loss,
                x0=init_s0,
                bounds=((self.config.s_min, self.config.init_s_max),),
                options={"maxiter": int(sum(count))},
            )
            params = res.x
            stability = params[0]
            rating_stability[int(first_rating)] = stability
            rating_count[int(first_rating)] = sum(count)

        for small_rating, big_rating in (
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 3),
            (2, 4),
            (1, 4),
        ):
            if small_rating in rating_stability and big_rating in rating_stability:
                # if rating_count[small_rating] > 300 and rating_count[big_rating] > 300:
                #     continue
                if rating_stability[small_rating] > rating_stability[big_rating]:
                    if rating_count[small_rating] > rating_count[big_rating]:
                        rating_stability[big_rating] = rating_stability[small_rating]
                    else:
                        rating_stability[small_rating] = rating_stability[big_rating]

        w1 = 0.41
        w2 = 0.54

        if len(rating_stability) == 0:
            raise Exception("Not enough data for pretraining!")
        elif len(rating_stability) == 1:
            rating = list(rating_stability.keys())[0]
            factor = rating_stability[rating] / r_s0_default[str(rating)]
            init_s0 = list(map(lambda x: x * factor, r_s0_default.values()))
        elif len(rating_stability) == 2:
            if 1 not in rating_stability and 2 not in rating_stability:
                rating_stability[2] = np.power(
                    rating_stability[3], 1 / (1 - w2)
                ) * np.power(rating_stability[4], 1 - 1 / (1 - w2))
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 1 not in rating_stability and 3 not in rating_stability:
                rating_stability[3] = np.power(rating_stability[2], 1 - w2) * np.power(
                    rating_stability[4], w2
                )
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 1 not in rating_stability and 4 not in rating_stability:
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 2 not in rating_stability and 3 not in rating_stability:
                rating_stability[2] = np.power(
                    rating_stability[1], w1 / (w1 + w2 - w1 * w2)
                ) * np.power(rating_stability[4], 1 - w1 / (w1 + w2 - w1 * w2))
                rating_stability[3] = np.power(
                    rating_stability[1], 1 - w2 / (w1 + w2 - w1 * w2)
                ) * np.power(rating_stability[4], w2 / (w1 + w2 - w1 * w2))
            elif 2 not in rating_stability and 4 not in rating_stability:
                rating_stability[2] = np.power(rating_stability[1], w1) * np.power(
                    rating_stability[3], 1 - w1
                )
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            elif 3 not in rating_stability and 4 not in rating_stability:
                rating_stability[3] = np.power(
                    rating_stability[1], 1 - 1 / (1 - w1)
                ) * np.power(rating_stability[2], 1 / (1 - w1))
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            init_s0 = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        elif len(rating_stability) == 3:
            if 1 not in rating_stability:
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 2 not in rating_stability:
                rating_stability[2] = np.power(rating_stability[1], w1) * np.power(
                    rating_stability[3], 1 - w1
                )
            elif 3 not in rating_stability:
                rating_stability[3] = np.power(rating_stability[2], 1 - w2) * np.power(
                    rating_stability[4], w2
                )
            elif 4 not in rating_stability:
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            init_s0 = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        elif len(rating_stability) == 4:
            init_s0 = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        self.w.data[0:4] = Tensor(
            list(
                map(
                    lambda x: max(min(self.config.init_s_max, x), self.config.s_min),
                    init_s0,
                )
            )
        )
        self.init_w_tensor = self.w.data.clone().to(self.config.device)
