class Constant:
    def __init__(self, start):
        self.val = start

    def anneal(self):
        pass


class LinearAnneal:
    """Linear Annealing Schedule.

    Args:
        start: The initial value of epsilon.
        end: The final value of epsilon.
        duration: The number of anneals from start value to end value.

    """

    def __init__(self, start: float, end: float, duration: int):
        self.val = start
        self.min = end
        self.duration = duration

    def anneal(self):
        self.val = max(self.min, self.val - (self.val - self.min) / self.duration)
