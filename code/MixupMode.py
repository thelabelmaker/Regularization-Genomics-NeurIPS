import enum

class MixupMode(enum.Enum):
    RANDOM_MIXUP = 'manifold_mixup'
    NO_MIXUP = 'no_mixup'
