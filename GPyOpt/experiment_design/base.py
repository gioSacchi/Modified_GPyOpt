class ExperimentDesign(object):
    """
    Base class for all experiment designs
    """
    def __init__(self, space, context=None):
        self.space = space
        self.context = context

    def get_samples(self, init_points_count):
        raise NotImplementedError("Subclasses should implement this method.")
