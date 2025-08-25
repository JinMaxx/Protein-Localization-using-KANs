import os
import pickle
import sys



figures_dir = "./data/figures"


# Set script to work from the project root directory
try: from source.abstract_figures import _AbstractFigure
except ImportError:
    print("Error: Could not import _AbstractFigure.")
    print("Please ensure the script is run from the project's root directory.")
    sys.exit(1)


def display_figure(file_path: str):
    """Loads and displays a Plotly figure from a .pkl file."""
    print(f"\nLoading and displaying: {os.path.basename(file_path)}...")
    try:
        with open(file_path, "rb") as f:
            # Load with pickle directly since the custom class might not be in the path,
            # but the underlying plotly Figure object is what we need.
            fig = pickle.load(f)

        # The .show() method is what renders the figure.
        fig.show()
        print("Figure window opened. Close it to continue.")
    except Exception as e:
        print(f"\n--- Error loading or displaying figure ---")
        print(f"File: {file_path}")
        print(f"Error: {e}")
        print("----------------------------------------")


def browse_directory(current_path: str):
    """The main interactive loop for browsing directories and selecting files."""

    while True:
        print("\n" + "=" * 50)
        print(f"Current Directory: {os.path.relpath(current_path, figures_dir)}")
        print("=" * 50)

        try:
            items = sorted(os.listdir(current_path))
        except FileNotFoundError:
            print(f"Error: Directory not found: {current_path}")
            return

        # Separate directories and .pkl files
        dirs = [d for d in items if os.path.isdir(os.path.join(current_path, d))]
        pkl_files = [f for f in items if f.endswith(".pkl")]

        # Display Menu
        print("Please make a selection:\n")

        # Display subdirectories
        for i, dirname in enumerate(dirs):
            print(f"  {i + 1:2d}) [DIR] {dirname}")

        dir_count = len(dirs)

        # Display .pkl files
        if pkl_files:
            print("-" * 20)
        for i, filename in enumerate(pkl_files):
            print(f"  {dir_count + i + 1:2d}) [FIG] {filename}")

        print("\n" + "-" * 20)
        if current_path != os.path.abspath(figures_dir):
            print("  b) Back")
        print("  q) Quit")

        # Get User Input
        choice = input("\nEnter your choice: ").strip().lower()

        if choice == 'q':
            break

        if choice == 'b' and current_path != os.path.abspath(figures_dir):
            current_path = os.path.dirname(current_path)
            continue

        try:
            choice_index = int(choice) - 1
            if not (0 <= choice_index < len(dirs) + len(pkl_files)):
                raise ValueError

            # If a directory was chosen
            if choice_index < len(dirs):
                current_path = os.path.join(current_path, dirs[choice_index])
            # If a .pkl file was chosen
            else:
                file_to_display = pkl_files[choice_index - len(dirs)]
                full_path = os.path.join(current_path, file_to_display)
                display_figure(full_path)

        except ValueError:
            print("\n*** Invalid choice. Please try again. ***")


def main():
    abs_figure_path = os.path.abspath(figures_dir)

    if not os.path.isdir(abs_figure_path):
        print(f"Error: figure directory '{figures_dir}' does not exist.")
        print("Please update the figures_dir variable in the script.")
        return

    print("--- Plotly Figure Browser ---")
    browse_directory(abs_figure_path)
    print("\nExiting browser. Goodbye!")


if __name__ == "__main__":
    main()