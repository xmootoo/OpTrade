import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Literal
import os
from datetime import datetime

from optparse import Values

# Custom modules
from optrade.data.thetadata.get_data import get_data

def set_theme(theme: Literal['light', 'dark'] = 'light'):
    """Set the plotting theme parameters."""
    base_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.titlesize": 20,
        "axes.labelsize": 18,  # Increased from 16
        "xtick.labelsize": 18,
        "ytick.labelsize": 16,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 1.5,
        "axes.grid": False,
    }

    if theme == 'dark':
        theme_params = {
            **base_params,
            "figure.facecolor": "black",
            "axes.facecolor": "black",
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        }
        plot_colors = {
            'face': 'black',
            'text': 'white',
            'line': 'white',
            'grid': 'white',
            'spine': 'white'
        }
    else:  # light theme
        theme_params = {
            **base_params,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
        plot_colors = {
            'face': 'white',
            'text': 'black',
            'line': 'black',
            'grid': 'gray',
            'spine': 'black'
        }

    plt.rcParams.update(theme_params)
    return plot_colors

def analyze_time_series(
    data_list: List[np.ndarray],
    info_list: List[Dict],
    dates: np.ndarray,
    volume_list: Optional[List[np.ndarray]] = None,
    dpi: int = 600,
    output_format: str = "eps",
    theme: Literal['light', 'dark'] = 'light',
    save_dir: Optional[str] = None
) -> None:
    """
    Analyze and visualize time series data with improved LaTeX formatting and larger font sizes.

    Args:
        data_list (List[np.ndarray]): List of time series datasets.
        info_list (List[Dict]): List of dictionaries containing information for each time series datasets.
        dates_list (List[np.ndarray]): List of date strings numpy arrays for each dataset. Default is None.
        dpi (int): DPI for saving plots. Default is 600.
        output_format (str): Output format for saving plots. Default is "eps".
        theme (str): Plot theme, either 'light' or 'dark'. Default is 'light'.
    """
    # Set theme and get colors
    colors = set_theme(theme)

    # Create output directory
    output_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', 'optrade', 'figures', 'general'))
    os.makedirs(output_dir, exist_ok=True)


    for i in range(len(data_list)):
        # Calculate global min and max for amplitude axis
        global_min = np.min(data_list[i])
        global_max = np.max(data_list[i])

        # Add some padding to the limits
        y_range = global_max - global_min
        global_min -= y_range * 0.05
        global_max += y_range * 0.05

        # Individual channel analysis
        data = data_list[i].squeeze()

        # Get title and y-axis label from info_list
        title = info_list[i]["title"]
        y_axis = info_list[i]["y-axis"]

        # Create figure
        fig = plt.figure(figsize=(24, 5), facecolor=colors['face'])
        ax = fig.add_subplot(111)

        # Original plotting code
        x_indices = list(range(len(dates)))
        ax.plot(x_indices, data, color=colors['line'], linewidth=1.2)

        # Title and labels
        ax.set_title(fr"\textbf{{{title}}}", fontsize=32, pad=20, color=colors['text'])
        ax.set_ylabel(r"\textrm{" + y_axis + r"}", fontsize=26, color='black')
        ax.set_ylim(global_min, global_max)

        # Set tick size for y-axis
        plt.yticks(fontsize=26)

        # # Set up ticks
        # num_ticks = 5
        # step = max(len(x_indices) // num_ticks, 1)
        # selected_indices = x_indices[::step]
        # selected_dates = [dates[i] for i in selected_indices]
        # ax.set_xticks(selected_indices)
        # ax.set_xticklabels(selected_dates, fontsize=26, color='black')

        # # Common styling for both cases
        # # Center align the date labels
        # for tick in ax.get_xticklabels():
        #     tick.set_ha('center')

        num_ticks = 5
        step = max(len(x_indices) // num_ticks, 1)
        selected_indices = x_indices[::step]
        selected_dates = [dates[i] for i in selected_indices]

        # Create axis with adjusted parameters
        ax.set_xticks(selected_indices)
        ax.set_xticklabels(selected_dates, fontsize=26, color='black', rotation=0)

        # Enhance bottom spine visibility
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(1.5)

        # Adjust margins to prevent cutoff
        plt.subplots_adjust(bottom=0.2, right=0.95)

        # Properly align tick labels
        for tick in ax.get_xticklabels():
            tick.set_ha('center')
            tick.set_va('top')


        # Style the spines
        for spine in ax.spines.values():
            spine.set_color(colors['spine'])
            spine.set_linewidth(1.5)

        # Add subtle grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.2, color=colors['grid'])

        # Adjust layout
        plt.subplots_adjust(bottom=0.2)

        # Save with tight layout
        figure_name = info_list[i]["file_codename"]
        plt.savefig(
            os.path.join(output_dir, f"{figure_name}.{output_format}"),
            dpi=dpi,
            bbox_inches='tight',
            facecolor=colors['face'],
            edgecolor='none'
        )
        plt.close()


def format_dates(dates):
    """
    Convert pandas datetime objects to 'MMM DD, YYYY (H:MMam)' format
    Example: Dec 06, 2023 (9:30am)

    Args:
        dates (pd.Series): Series of pandas datetime objects
    Returns:
        np.ndarray: Formatted date strings
    """
    # Create initial format with leading zeros
    formatted_dates = [d.strftime('%b %d, %Y (%I:%M%p)').replace(' 0', ' ').lower() for d in dates]

    # Capitalize month and handle day leading zeros
    capitalized = []
    for d in formatted_dates:
        # Split into parts
        month_part = d[:3].capitalize()  # First 3 chars are month
        rest = d[3:]  # Rest of the string

        # If day starts with 0, remove it (but keep it for single digit days)
        if rest.startswith(' 0'):
            rest = ' ' + rest[2:]

        capitalized.append(month_part + rest)

    return np.array(capitalized)


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Time Series Analysis")
    parser.add_argument("root", type=str, default="AAPL", help="Root of security to analyze.")
    parser.add_argument("mode", type=str, default="option", help="Whether to analyze option or underlying data. Options: 'option', 'stock', or 'both'.")
    parser.add_argument("start_date", type=str, default="20241107", help="Start date for data in YYYYMMDD format.")
    parser.add_argument("end_date", type=str, default="20241107", help="End date for data in YYYYMMDD format.")
    parser.add_argument("exp", type=str, default="20250117", help="Expiration date for option in YYYYMMDD format.")
    parser.add_argument("strike", type=int, default=225, help="Strike price for option.")
    parser.add_argument("right", type=str, default="C", help="Option type, either 'C' for call or 'P' for put.")
    parser.add_argument("output_format", type=str, default="eps", help="Output format for saving plots. Default is 'eps'.")
    parser.add_argument("--volume", type=bool, default=False, help="Whether to include volume data in analysis.")
    args = parser.parse_args()

    # Set plot parameters
    right_name = "Call" if args.right == "C" else "Put"
    map = {"option":
                {
                    "title": args.root,
                    "codename": args.root,
                    "file_codename": f"{args.root}_{args.right}_{args.exp}_{args.strike}",
                    "y-axis": f"{right_name} Option Midprice",
                    "x-axis": "Date"
                },
            "stock":
                {
                    "title": args.root,
                    "codename": args.root,
                    "file_codename": f"{args.root}_stock",
                    "y-axis": "Underlying Midprice",
                    "x-axis": "Date"
                },
        }

    df = get_data(
        root=args.root,
        start_date=args.start_date,
        end_date=args.end_date,
        exp=args.exp,
        strike=args.strike,
        interval_min=1,
        right=args.right,
        save_dir="../../data/historical_data/merged")

    print(df.tail())

    data_list = []
    volume_list = []
    info_list = []

    if args.mode in ["option", "both"]:
        info_list.append(map["option"])
        option_mid = df["option_mid_price"].values
        data_list.append(option_mid)
        if args.volume:
            volume_list.append(df["option_volume"].values)

    if args.mode in ["stock", "both"]:
        info_list.append(map["stock"])
        stock_mid = df["stock_mid_price"].values
        data_list.append(stock_mid)
        if args.volume:
            volume_list.append(df["stock_volume"].values)

    dates = df["datetime"].values

    # In your main code, replace the ... with:
    dates = format_dates(df["datetime"])

    analyze_time_series(
        data_list=data_list,
        info_list=info_list,
        dates=dates,
        volume_list=volume_list,
        dpi = 600,
        output_format=args.output_format,
        )
