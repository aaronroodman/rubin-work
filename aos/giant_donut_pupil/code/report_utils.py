"""Shared report helpers for the giant-donut pupil passes.

Rule for every report page: a panel with nothing to show prints the reason on the
panel (`empty_panel`), so a blank panel is never ambiguous.
"""

__all__ = ['empty_panel']


def empty_panel(ax, reason, fontsize=8):
    """Blank `ax` and print `reason` in its centre."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, 'EMPTY: ' + reason, transform=ax.transAxes, ha='center', va='center',
            fontsize=fontsize, color='tab:red', wrap=True)
    return None
