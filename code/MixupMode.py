import enum

class MixupMode(enum.Enum):
    MANIFOLD_MIXUP = 'manifold_mixup'
    NO_MIXUP = 'no_mixup'
    RANDOM_MIXUP = 'random_mixup'
    STATIC_MIXUP = 'static_mixup'
    INPUT_MIXUP = 'input_mixup'
    GAUSSIAN_NOISE = 'gaussian_noise'
    RC = 'reverse_complement'
    ONLY_MIXUP = 'only_mixup'
    TARGETED_MIXUP = 'targeted_mixup'
