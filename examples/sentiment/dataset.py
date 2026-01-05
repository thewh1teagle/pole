"""Sentiment analysis datasets for testing prompt optimization."""

# Small dataset (7 examples) - Easy cases
SMALL_DATASET = [
    ("This movie was absolutely amazing!", "positive"),
    ("Worst experience ever, total waste of time.", "negative"),
    ("The product works fine, nothing special.", "neutral"),
    ("I love this so much! Best purchase ever!", "positive"),
    ("Terrible quality, very disappointed.", "negative"),
    ("It's okay, does what it says.", "neutral"),
    ("Incredible! Exceeded all my expectations!", "positive"),
]

# Medium dataset (15 examples) - Mix of clear and subtle cases
MEDIUM_DATASET = [
    # Clear positive
    ("This movie was absolutely amazing!", "positive"),
    ("I love this so much! Best purchase ever!", "positive"),
    ("Incredible! Exceeded all my expectations!", "positive"),

    # Clear negative
    ("Worst experience ever, total waste of time.", "negative"),
    ("Terrible quality, very disappointed.", "negative"),
    ("Awful service, would not recommend.", "negative"),

    # Clear neutral
    ("The product works fine, nothing special.", "neutral"),
    ("It's okay, does what it says.", "neutral"),
    ("Average experience, nothing to complain about.", "neutral"),

    # Subtle positive
    ("Not bad at all, actually pretty good.", "positive"),
    ("Better than expected, quite satisfied.", "positive"),

    # Subtle negative
    ("Could have been better, somewhat disappointing.", "negative"),
    ("Not what I hoped for, mediocre at best.", "negative"),

    # Tricky neutral
    ("It works, that's about it.", "neutral"),
    ("Does the job, no complaints.", "neutral"),
]

# Large challenging dataset (30 examples) - Includes sarcasm, mixed sentiment, edge cases
CHALLENGING_DATASET = [
    # Clear cases
    ("This movie was absolutely amazing!", "positive"),
    ("Worst experience ever, total waste of time.", "negative"),
    ("The product works fine, nothing special.", "neutral"),
    ("I love this so much! Best purchase ever!", "positive"),
    ("Terrible quality, very disappointed.", "negative"),
    ("It's okay, does what it says.", "neutral"),

    # Sarcasm (challenging)
    ("Oh great, another delay. Just what I needed.", "negative"),
    ("Yeah, because waiting 3 hours is totally reasonable.", "negative"),

    # Mixed sentiment (lean one way)
    ("The food was great but the service was terrible.", "neutral"),
    ("Amazing product, shame about the price.", "positive"),
    ("Fast delivery but product arrived damaged.", "negative"),

    # Subtle sentiment
    ("Not bad at all, actually pretty good.", "positive"),
    ("Could have been better, somewhat disappointing.", "negative"),
    ("It works, that's about it.", "neutral"),
    ("Better than expected, quite satisfied.", "positive"),
    ("Not what I hoped for, mediocre at best.", "negative"),

    # Double negatives
    ("I wouldn't say it's not good.", "positive"),
    ("Can't complain, not disappointed.", "positive"),

    # Understatement
    ("It's not the worst thing ever.", "neutral"),
    ("I've seen worse.", "neutral"),

    # Enthusiastic positive
    ("Absolutely phenomenal! 10/10 would recommend!", "positive"),
    ("Best thing I've ever bought, hands down!", "positive"),

    # Strong negative
    ("Complete disaster, avoid at all costs!", "negative"),
    ("Garbage product, waste of money!", "negative"),

    # Professional neutral
    ("The service meets industry standards.", "neutral"),
    ("Adequate performance for the price point.", "neutral"),
    ("Functions as described in documentation.", "neutral"),

    # Edge cases with numbers/stats
    ("8/10, pretty solid overall.", "positive"),
    ("2/10, would not buy again.", "negative"),
    ("5/10, nothing special.", "neutral"),
]

# Helper function to get dataset by name
def get_dataset(name: str = "small"):
    """
    Get dataset by name.

    Args:
        name: One of "small", "medium", or "challenging"

    Returns:
        List of (text, sentiment) tuples
    """
    datasets = {
        "small": SMALL_DATASET,
        "medium": MEDIUM_DATASET,
        "challenging": CHALLENGING_DATASET,
    }

    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(datasets.keys())}")

    return datasets[name]
