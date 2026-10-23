"""Plain-language decision support; no claims of validated machine safety."""
import math
from .data import risk_state

RUL_EXPLANATION = ("Remaining useful life is the model's estimate of how many operating cycles may remain "
                   "before the system reaches its learned end-of-life condition.")
GUIDANCE_LABEL = 'Decision-support guidance only — not a validated operational procedure.'


def operator_summary(prediction, high=30., elevated=100.):
    state = risk_state(prediction, high, elevated)
    level = state['level']
    messages = {
        'invalid': ('Unable to assess', '⊘',
                    'Check the input data and ask a qualified technician to review it. Do not rely on this estimate.',
                    'The model did not provide a usable, non-negative estimate.'),
        'high': ('Needs review', '⚠',
                 'Schedule a maintenance inspection and ask a qualified technician to verify the condition.',
                 f'The estimate is at or below the illustrative review threshold of {high:g} cycles.'),
        'elevated': ('Monitor', '◷',
                     'Review the estimate and continue routine monitoring. Plan an inspection if concerns persist.',
                     f'The estimate is above {high:g} and at or below {elevated:g} cycles, within the illustrative monitoring band.'),
        'neutral': ('Within demonstration range', 'ⓘ',
                    'Continue routine monitoring and scheduled inspections. This estimate is not a safety clearance.',
                    f'The estimate is above the illustrative monitoring threshold of {elevated:g} cycles. This does not establish safety.'),
    }
    status, icon, action, reason = messages[level]
    valid = level != 'invalid'
    return {'status': status, 'icon': icon, 'action': action, 'reason': reason, 'level': level,
            'raw_prediction': prediction,
            'display_prediction': float(prediction) if valid else None,
            'formatted_cycles': f'{float(prediction):,.2f}' if valid else 'Unavailable'}


def operating_hours(prediction, rate=None):
    """Assumed operating hours, only for valid RUL and a supplied positive rate."""
    if rate is None:
        return None
    if isinstance(rate, bool) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('Enter a finite, positive verified operating rate in cycles per hour.')
    if prediction is None or not math.isfinite(prediction) or prediction < 0:
        return None
    hours = prediction / rate
    if not math.isfinite(hours):
        raise ValueError('This rate produces an unrepresentable hours estimate. Check the operating rate.')
    return hours
