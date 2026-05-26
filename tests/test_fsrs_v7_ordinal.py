from types import SimpleNamespace
import unittest

import torch

from models.fsrs_v7_ordinal import FSRS7Ordinal


def _config():
    return SimpleNamespace(
        device=torch.device("cpu"),
        s_min=0.0001,
        init_s_max=100.0,
        use_secs_intervals=True,
        sched_penalties=False,
    )


class FSRS7OrdinalTest(unittest.TestCase):
    def test_ordinal_probabilities_are_valid_distribution(self):
        model = FSRS7Ordinal(_config())
        retentions = torch.tensor([0.2, 0.5, 0.8])
        stabilities = torch.tensor([0.5, 2.0, 10.0])
        difficulties = torch.tensor([8.0, 5.0, 2.0])

        probabilities = model.ordinal_button_probabilities(
            retentions, stabilities, difficulties
        )
        h, e = model.ordinal_gates(retentions, stabilities, difficulties)

        self.assertTrue(torch.all(probabilities >= 0.0))
        self.assertTrue(
            torch.allclose(probabilities.sum(dim=1), torch.ones(3), atol=1e-6)
        )
        self.assertTrue(torch.all(e <= h))

    def test_ordinal_nll_uses_observed_button_probability(self):
        probabilities = torch.tensor(
            [
                [0.6, 0.2, 0.15, 0.05],
                [0.1, 0.5, 0.3, 0.1],
                [0.1, 0.2, 0.4, 0.3],
                [0.05, 0.15, 0.3, 0.5],
            ]
        )
        ratings = torch.tensor([1, 2, 3, 4])
        weights = torch.ones(4)

        expected = -torch.log(torch.tensor([0.6, 0.5, 0.4, 0.5])).sum()

        self.assertTrue(
            torch.allclose(
                FSRS7Ordinal.ordinal_nll(probabilities, ratings, weights), expected
            )
        )


if __name__ == "__main__":
    unittest.main()
