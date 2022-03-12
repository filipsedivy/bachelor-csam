from application import decode_predictions_dictionary
from enum import Enum


class ScoringResult(Enum):
    NOT_CSAM = 1
    MAYBE_CSAM = 2
    SURE_CSAM = 3


def is_csam(preds) -> ScoringResult:
    result = decode_predictions_dictionary(preds)

    # Child naked >= 90%
    if result['child_naked'] >= 0.9:
        return ScoringResult.SURE_CSAM

    # Child >= 10% && Child naked >= 40%
    if result['child'] >= 0.1 and result['child_naked'] >= 0.4:
        return ScoringResult.MAYBE_CSAM

    # Child naked >= 50% && Adult naked <= 0,8%
    if result['child_naked'] >= 0.5 and result['adult_naked'] <= 0.008:
        return ScoringResult.SURE_CSAM

    return ScoringResult.NOT_CSAM
