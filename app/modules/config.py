# Configuration constants

# max number of samples to retain
MAX_SAMPLES = 1000
# secs between samples
SAMPLE_PERIOD = 1

SYSTEM_PREFIXES = {
        "wexac": "/home/pedro/wexac/",
        "labs": "/home/labs/nyosef/pedro/",
        "laptop": "/Volumes/pedro",
        }

# Theme settings
DARK_TEMPLATE = dict(
        layout=dict(
            paper_bgcolor='#2d3339',  # Match Slate theme background
            plot_bgcolor='#2d3339',
            font=dict(color='#ffffff'),
            xaxis=dict(
                gridcolor='#4f5b66',
                linecolor='#4f5b66',
                zerolinecolor='#4f5b66'
                ),
            yaxis=dict(
                gridcolor='#4f5b66',
                linecolor='#4f5b66',
                zerolinecolor='#4f5b66'
                ),
            margin=dict(t=30, l=10, r=10, b=10)
            )
        )

# Sequence color settings
SEQUENCE_COLORS = {
    'GAGACTGCATGG': '#50C878',  # Emerald green for theme
    'TTTAGTGAGGGT': '#9370DB'   # Medium purple for theme
}
