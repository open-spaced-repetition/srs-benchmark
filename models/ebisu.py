import ebisu  # type: ignore


class Ebisu:
    def ebisu_v2(self, sequence):
        """
        Implementation of the Ebisu v2 algorithm.

        Args:
            sequence: A sequence of (delta_t, rating) tuples

        Returns:
            The trained Ebisu model
        """
        init_ivl = 512
        alpha = 0.2
        beta = 0.2
        model = ebisu.defaultModel(init_ivl, alpha, beta)
        for delta_t, rating in sequence:
            model = ebisu.updateRecall(
                model,
                successes=1 if rating > 1 else 0,
                total=1,
                tnow=max(delta_t, 0.001),
            )
        return model

    def predict(self, model, delta_t):
        return ebisu.predictRecall(model, tnow=max(delta_t, 0.001), exact=True)
