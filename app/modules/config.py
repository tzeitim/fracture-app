# Configuration constants

# max number of samples to retain
MAX_SAMPLES = 1000
# secs between samples
SAMPLE_PERIOD = 1

SYSTEM_PREFIXES = {
        "wexac": "/home/pedro/wexac/",
        "labs": "/home/projects/nyosef/pedro/",
        "laptop": "/Volumes/pedro",
        }

# Theme settings
# Current dark theme (slate)
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

# Catppuccin Latte (light theme)
LATTE_TEMPLATE = dict(
        layout=dict(
            paper_bgcolor='#eff1f5',  # Base
            plot_bgcolor='#e6e9ef',   # Mantle
            font=dict(color='#4c4f69'),  # Text
            xaxis=dict(
                gridcolor='#ccd0da',  # Surface1
                linecolor='#ccd0da',
                zerolinecolor='#ccd0da'
                ),
            yaxis=dict(
                gridcolor='#ccd0da',
                linecolor='#ccd0da',
                zerolinecolor='#ccd0da'
                ),
            margin=dict(t=30, l=10, r=10, b=10)
            )
        )

# Catppuccin Mocha (dark theme)
MOCHA_TEMPLATE = dict(
        layout=dict(
            paper_bgcolor='#1e1e2e',  # Base
            plot_bgcolor='#181825',   # Mantle
            font=dict(color='#cdd6f4'),  # Text
            xaxis=dict(
                gridcolor='#313244',  # Surface1
                linecolor='#313244',
                zerolinecolor='#313244'
                ),
            yaxis=dict(
                gridcolor='#313244',
                linecolor='#313244',
                zerolinecolor='#313244'
                ),
            margin=dict(t=30, l=10, r=10, b=10)
            )
        )

# Sequence color settings
SEQUENCE_COLORS = {
    'GAGACTGCATGG': '#50C878',  # Emerald green for theme
    'TTTAGTGAGGGT': '#9370DB'   # Medium purple for theme
}
