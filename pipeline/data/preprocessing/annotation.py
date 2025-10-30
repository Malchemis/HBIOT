"""Annotation processing for MEG data.

This module provides utilities for extracting and filtering spike annotations
from MEG data based on configurable pattern matching rules.
"""

import re
from typing import Dict, List, Any, Optional


def compile_annotation_patterns(annotation_rules: Dict[str, Any]) -> Dict[str, Dict[str, List[re.Pattern]]]:
    """Precompile regex patterns for annotation processing.

    Args:
        annotation_rules: Dictionary mapping group names to their annotation rules.
                         Each rule contains 'include' and 'exclude' pattern lists.

    Returns:
        Dictionary mapping groups to compiled regex patterns for include/exclude rules.
    """
    return {
        group: {
            'include': [re.compile(p.lower()) for p in rules.get('include', [])],
            'exclude': [re.compile(p.lower()) for p in rules.get('exclude', [])]
        }
        for group, rules in annotation_rules.items()
    }


def get_spike_annotations(annotations, group: str, compiled_patterns: Dict[str, Dict[str, List[re.Pattern]]]) -> List[float]:
    """Extract spike annotations based on the group's annotation rules.

    Args:
        annotations: The MNE annotations object.
        group: The dataset group (Holdout or IterativeLearningFeedback*).
        compiled_patterns: Precompiled regex patterns for annotation processing.

    Returns:
        List of spike onset times in seconds.
    """
    descriptions = annotations.description
    onsets = annotations.onset

    group_key = group if group in compiled_patterns else 'Default'
    patterns = compiled_patterns[group_key]
    include_regex = patterns['include']
    exclude_regex = patterns['exclude']

    spike_onsets = []
    for i in range(len(annotations)):
        description = descriptions[i].lower()
        onset = onsets[i]

        included = False
        if len(include_regex) == 0:
            included = True
        else:
            for pattern in include_regex:
                if pattern.search(description):
                    included = True
                    break

        excluded = False
        for pattern in exclude_regex:
            if pattern.search(description):
                excluded = True
                break

        if included and not excluded:
            spike_onsets.append(onset)

    return spike_onsets
