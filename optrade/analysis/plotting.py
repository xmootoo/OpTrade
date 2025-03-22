import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Literal
import os
from datetime import datetime

# Custom modules
from optrade.data.thetadata import load_all_data


def set_theme(theme: Literal["light", "dark"] = "light"):
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

    if theme == "dark":
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
            "face": "black",
            "text": "white",
            "line": "white",
            "grid": "white",
            "spine": "white",
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
            "face": "white",
            "text": "black",
            "line": "black",
            "grid": "gray",
            "spine": "black",
        }

    plt.rcParams.update(theme_params)
    return plot_colors


# def analyze_time_series(
#     data_list: List[np.ndarray],
#     info_list: List[Dict],
#     dates: np.ndarray,
#     dpi: int = 600,
#     output_format: str = "eps",
#     theme: Literal['light', 'dark'] = 'light',
#     save_dir: Optional[str] = None,
#     volume: bool = False,
#     volume_list: Optional[List[np.ndarray]] = None,
# ) -> None:
#     """
#     Analyze and visualize time series data with improved LaTeX formatting and larger font sizes.

#     Args:
#         data_list (List[np.ndarray]): List of time series datasets.
#         info_list (List[Dict]): List of dictionaries containing information for each time series datasets.
#         dates_list (List[np.ndarray]): List of date strings numpy arrays for each dataset. Default is None.
#         dpi (int): DPI for saving plots. Default is 600.
#         output_format (str): Output format for saving plots. Default is "eps".
#         theme (str): Plot theme, either 'light' or 'dark'. Default is 'light'.
#     """
#     # Set theme and get colors
#     colors = set_theme(theme)

#     # Create output directory
#     output_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', 'optrade', 'figures', 'general'))
#     os.makedirs(output_dir, exist_ok=True)


#     for i in range(len(data_list)):
#         # Calculate global min and max for amplitude axis
#         global_min = np.min(data_list[i])
#         global_max = np.max(data_list[i])

#         # Add some padding to the limits
#         y_range = global_max - global_min
#         global_min -= y_range * 0.05
#         global_max += y_range * 0.05

#         # Individual channel analysis
#         data = data_list[i].squeeze()

#         # Get title and y-axis label from info_list
#         title = info_list[i]["title"]
#         y_axis = info_list[i]["y-axis"]

#         # Create figure
#         fig = plt.figure(figsize=(12, 8), facecolor=colors['face'])
#         ax = fig.add_subplot(111)

#         # Original plotting code
#         x_indices = list(range(len(dates)))
#         ax.plot(x_indices, data, color=colors['line'], linewidth=1.2)

#         # Title and labels
#         ax.set_title(fr"\textbf{{{title}}}", fontsize=32, pad=20, color=colors['text'])
#         ax.set_ylabel(r"\textrm{" + y_axis + r"}", fontsize=26, color='black')
#         ax.set_ylim(global_min, global_max)

#         # Set tick size for y-axis
#         plt.yticks(fontsize=26)

#         # # Set up ticks
#         # num_ticks = 5
#         # step = max(len(x_indices) // num_ticks, 1)
#         # selected_indices = x_indices[::step]
#         # selected_dates = [dates[i] for i in selected_indices]
#         # ax.set_xticks(selected_indices)
#         # ax.set_xticklabels(selected_dates, fontsize=26, color='black')

#         # # Common styling for both cases
#         # # Center align the date labels
#         # for tick in ax.get_xticklabels():
#         #     tick.set_ha('center')

#         num_ticks = 5
#         step = max(len(x_indices) // num_ticks, 1)
#         selected_indices = x_indices[::step]
#         selected_dates = [dates[i] for i in selected_indices]

#         # Create axis with adjusted parameters
#         ax.set_xticks(selected_indices)
#         ax.set_xticklabels(selected_dates, fontsize=26, color='black', rotation=0)

#         # Enhance bottom spine visibility
#         ax.spines['bottom'].set_color('black')
#         ax.spines['bottom'].set_linewidth(1.5)

#         # Adjust margins to prevent cutoff
#         plt.subplots_adjust(bottom=0.2, right=0.95)

#         # Properly align tick labels
#         for tick in ax.get_xticklabels():
#             tick.set_ha('center')
#             tick.set_va('top')


#         # Style the spines
#         for spine in ax.spines.values():
#             spine.set_color(colors['spine'])
#             spine.set_linewidth(1.5)

#         # Add subtle grid
#         ax.grid(True, axis='y', linestyle='--', alpha=0.2, color=colors['grid'])

#         # Adjust layout
#         plt.subplots_adjust(bottom=0.2)

#         # Save with tight layout
#         figure_name = info_list[i]["file_codename"]
#         plt.savefig(
#             os.path.join(output_dir, f"{figure_name}.{output_format}"),
#             dpi=dpi,
#             bbox_inches='tight',
#             facecolor=colors['face'],
#             edgecolor='none'
#         )
#         plt.close()


def analyze_time_series(
    data_list: List[np.ndarray],
    info_list: List[Dict],
    dates: np.ndarray,
    dpi: int = 600,
    output_format: str = "eps",
    theme: Literal["light", "dark"] = "light",
    save_dir: Optional[str] = None,
    volume: bool = True,
    volume_list: Optional[List[np.ndarray]] = None,
) -> None:
    """
    Analyze and visualize time series data with normalized volume subplot.
    """
    from sklearn.preprocessing import StandardScaler

    # Set theme and get colors
    colors = set_theme(theme)

    # Create output directory
    output_dir = os.path.abspath(
        os.path.join(os.getcwd(), "..", "..", "..", "optrade", "figures", "general")
    )
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(data_list)):
        # Normalize volume data
        if volume and volume_list is not None:
            volume_data = volume_list[i].squeeze()

            # Print original volume stats
            print(f"Original volume stats:")
            print(f"Min: {np.min(volume_data)}")
            print(f"Max: {np.max(volume_data)}")
            print(f"Mean: {np.mean(volume_data)}")

            # Reshape for StandardScaler
            volume_data_reshaped = volume_data.reshape(-1, 1)
            scaler = StandardScaler()
            volume_data_normalized = scaler.fit_transform(
                volume_data_reshaped
            ).squeeze()

            # Print normalized volume stats
            print(f"\nNormalized volume stats:")
            print(f"Min: {np.min(volume_data_normalized)}")
            print(f"Max: {np.max(volume_data_normalized)}")
            print(f"Mean: {np.mean(volume_data_normalized)}")

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

        # Create figure with subplots
        if volume and volume_list is not None:
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(12, 10),
                gridspec_kw={"height_ratios": [3, 1]},
                facecolor=colors["face"],
            )
            fig.subplots_adjust(hspace=0)
        else:
            fig = plt.figure(figsize=(12, 8), facecolor=colors["face"])
            ax1 = fig.add_subplot(111)

        # Plot price data
        x_indices = list(range(len(dates)))
        ax1.plot(x_indices, data, color=colors["line"], linewidth=1.2)

        # Title and labels for price subplot
        ax1.set_title(rf"\textbf{{{title}}}", fontsize=32, pad=20, color=colors["text"])
        ax1.set_ylabel(r"\textrm{" + y_axis + r"}", fontsize=26, color="black")
        ax1.set_ylim(global_min, global_max)

        # Set tick size for y-axis
        ax1.tick_params(axis="y", labelsize=26)

        if volume and volume_list is not None:
            # Remove x-axis labels from price subplot
            ax1.set_xticklabels([])
            ax1.set_xticks([])

            # Plot normalized volume bars
            ax2.bar(
                x_indices,
                volume_data_normalized,
                color="royalblue",
                alpha=1.0,
                width=1.0,
            )

            # Set y-axis limits for normalized volume
            ax2.set_ylim(
                min(volume_data_normalized) * 1.1, max(volume_data_normalized) * 1.1
            )

            # Style volume subplot
            ax2.set_ylabel(
                r"\textrm{Normalized Volume ($\sigma$)}", fontsize=26, color="black"
            )
            ax2.tick_params(axis="y", labelsize=26)

            # Remove x-axis
            ax2.set_xticklabels([])
            ax2.set_xticks([])

            # Style volume subplot spines
            for spine in ax2.spines.values():
                spine.set_color(colors["spine"])
                spine.set_linewidth(1.5)

            # Make grid more visible
            ax2.grid(True, axis="y", linestyle="-", alpha=0.3, color="gray")

            plt.subplots_adjust(bottom=0.1)
        else:
            ax1.set_xticklabels([])
            ax1.set_xticks([])

        # Style the price subplot spines
        for spine in ax1.spines.values():
            spine.set_color(colors["spine"])
            spine.set_linewidth(1.5)

        # Add subtle grid to price subplot
        ax1.grid(True, axis="y", linestyle="--", alpha=0.2, color=colors["grid"])

        # Save with tight layout
        figure_name = info_list[i]["file_codename"]
        plt.savefig(
            os.path.join(output_dir, f"{figure_name}.{output_format}"),
            dpi=dpi,
            bbox_inches="tight",
            facecolor=colors["face"],
            edgecolor="none",
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
    formatted_dates = [
        d.strftime("%b %d, %Y (%I:%M%p)").replace(" 0", " ").lower() for d in dates
    ]

    # Capitalize month and handle day leading zeros
    capitalized = []
    for d in formatted_dates:
        # Split into parts
        month_part = d[:3].capitalize()  # First 3 chars are month
        rest = d[3:]  # Rest of the string

        # If day starts with 0, remove it (but keep it for single digit days)
        if rest.startswith(" 0"):
            rest = " " + rest[2:]

        capitalized.append(month_part + rest)

    return np.array(capitalized)


def plot_returns_distribution(data: np.ndarray, log: bool = False):
    """
    Plot the distribution of (optionally log) returns for a given security.
    """


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Time Series Analysis")
    parser.add_argument(
        "root", type=str, default="AAPL", help="Root of security to analyze."
    )
    parser.add_argument(
        "mode",
        type=str,
        default="option",
        help="Whether to analyze option or underlying data. Options: 'option', 'stock', or 'both'.",
    )
    parser.add_argument(
        "start_date",
        type=str,
        default="20241107",
        help="Start date for data in YYYYMMDD format.",
    )
    parser.add_argument(
        "end_date",
        type=str,
        default="20241107",
        help="End date for data in YYYYMMDD format.",
    )
    parser.add_argument(
        "exp",
        type=str,
        default="20250117",
        help="Expiration date for option in YYYYMMDD format.",
    )
    parser.add_argument(
        "strike", type=int, default=225, help="Strike price for option."
    )
    parser.add_argument(
        "right",
        type=str,
        default="C",
        help="Option type, either 'C' for call or 'P' for put.",
    )
    parser.add_argument(
        "output_format",
        type=str,
        default="eps",
        help="Output format for saving plots. Default is 'eps'.",
    )
    parser.add_argument(
        "--volume", action="store_true", help="Include volume data in analysis."
    )
    args = parser.parse_args()

    # Set plot parameters
    right_name = "Call" if args.right == "C" else "Put"
    map = {
        "option": {
            "title": args.root,
            "codename": args.root,
            "file_codename": f"{args.root}_{args.right}_{args.exp}_{args.strike}",
            "y-axis": f"{right_name} Option Midprice",
            "x-axis": "Date",
        },
        "stock": {
            "title": args.root,
            "codename": args.root,
            "file_codename": f"{args.root}_stock",
            "y-axis": "Underlying Midprice",
            "x-axis": "Date",
        },
    }

    from optrade.data.thetadata.contracts import Contract

    contract = Contract(
        root=args.root,
        start_date=args.start_date,
        exp=args.exp,
        strike=args.strike,
        interval_min=1,
        right=args.right,
    )

    df = load_all_data(contract=contract, clean_up=True)

    # Clip data to end datetime
    end_datetime = datetime.strptime(args.end_date, "%Y%m%d")
    df = df[df["datetime"] <= end_datetime]

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

    volume_list = []
    if args.volume:
        if args.mode in ["option", "both"]:
            volume_list.append(df["option_volume"].values)
        if args.mode in ["stock", "both"]:
            volume_list.append(df["stock_volume"].values)

    analyze_time_series(
        data_list=data_list,
        info_list=info_list,
        dates=dates,
        volume_list=None,
        dpi=600,
        output_format=args.output_format,
    )
