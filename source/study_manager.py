import os
import sys
import optuna
import pandas as pd
from typing import Optional



studies_dir = "./data/studies"

# Increase pandas display width for better readability of dataframes
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)


def load_study(study_path: str) -> Optional[optuna.Study]:
    """Safely loads an Optuna study from an SQLite .db file."""
    study_name = os.path.basename(study_path).replace(".db", "")
    storage_url = f"sqlite:///{study_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        print(f"\nSuccessfully loaded study: {study.study_name}")
        return study
    except KeyError:
        print(f"Error: No study named '{study_name}' found in the database file '{os.path.basename(study_path)}'.")
        print("This can happen if the filename does not match the study_name inside it.")
    except Exception as e:
        print(f"An unexpected error occurred while loading the study: {e}")
    return None


def view_study_summary(study: optuna.Study):
    """Prints a summary of the study's trials and best result."""
    print("\n--- Study Summary ---")
    print(f"Study Name: {study.study_name}")
    print(f"Direction: {study.direction.name}")
    print(f"Number of trials: {len(study.trials)}")

    try:
        print("\n--- Best Trial ---")
        best = study.best_trial
        print(f"  Number: {best.number}")
        print(f"  Value: {best.value:.6f}")
        print("  Params: ")
        for key, value in best.params.items():
            print(f"    {key}: {value}")
    except ValueError:
        print("No completed trials found in this study yet.")

    print("\n--- All Trials ---")
    print(study.trials_dataframe())
    print("-" * 20)


def delete_trial_from_db(study: optuna.Study):
    """Interactively and permanently deletes a trial from the study's database."""
    view_study_summary(study)
    try:
        trial_number_str = input("Enter the number of the trial to DELETE (or 'c' to cancel): ").strip().lower()
        if trial_number_str == 'c':
            return

        trial_number = int(trial_number_str)

        # Find the internal trial ID for the given trial number
        trial_to_delete = next((t for t in study.trials if t.number == trial_number), None)

        if trial_to_delete is None:
            print(f"Error: Trial number {trial_number} not found.")
            return

        confirm = input(
            f"WARNING: This will PERMANENTLY delete Trial #{trial_number} from the database.\nThis action cannot be undone. Are you sure? (y/n): ").lower()

        if confirm == 'y':
            # noinspection PyProtectedMember
            study._storage.delete_trial(trial_to_delete._trial_id)
            print(f"Trial #{trial_number} has been permanently deleted from the database.")
        else:
            print("Deletion cancelled.")

    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"An error occurred during deletion: {e}")


def manage_study_menu(study_path: str):
    """Main interactive menu for managing a single selected study."""
    while True:  # Loop to allow reloading after deletion
        study = load_study(study_path)
        if not study:
            input("Press Enter to return to file browser...")
            return

        print("\n" + "=" * 50)
        print(f"Managing Study: {study.study_name}")
        print(f"File: {os.path.basename(study_path)}")
        print("=" * 50)
        print("1) View summary")
        print("2) Delete a trial (Permanent)")
        print("3) Back to file browser")

        choice = input("> ").strip()

        if choice == '1':
            view_study_summary(study)
            input("\nPress Enter to continue...")
        elif choice == '2':
            delete_trial_from_db(study)
            # The study object is now stale, so the loop will reload it.
        elif choice == '3':
            break
        else:
            print("Invalid choice.")


def browse_studies(current_path: str):
    """The main interactive loop for browsing directories and selecting studies."""
    while True:
        print("\n" + "=" * 60)
        print(f"Current Directory: {os.path.relpath(current_path, studies_dir)}")
        print("=" * 60)

        try:
            items = sorted(os.listdir(current_path))
        except FileNotFoundError:
            print(f"Error: Directory not found: {current_path}")
            return

        dirs = [d for d in items if os.path.isdir(os.path.join(current_path, d))]
        db_files = [f for f in items if f.endswith(".db")]

        print("Please make a selection:\n")
        for i, dirname in enumerate(dirs):
            print(f"  {i + 1:2d}) [DIR]   {dirname}")

        if db_files:
            print("-" * 20)
        for i, filename in enumerate(db_files):
            print(f"  {len(dirs) + i + 1:2d}) [STUDY] {filename}")

        print("\n" + "-" * 20)
        if current_path != os.path.abspath(studies_dir):
            print("  b) Back")
        print("  q) Quit")

        choice = input("\nEnter your choice: ").strip().lower()

        if choice == 'q':
            break
        elif choice == 'b' and current_path != os.path.abspath(studies_dir):
            current_path = os.path.dirname(current_path)
            continue

        try:
            choice_index = int(choice) - 1
            if not (0 <= choice_index < len(dirs) + len(db_files)):
                raise ValueError

            if choice_index < len(dirs):
                current_path = os.path.join(current_path, dirs[choice_index])
            else:
                file_to_manage = db_files[choice_index - len(dirs)]
                full_path = os.path.join(current_path, file_to_manage)
                manage_study_menu(full_path)
        except ValueError:
            print("\n*** Invalid choice. Please try again. ***")


def main():
    """Main entry point for the study browser tool."""
    abs_studies_path = os.path.abspath(studies_dir)
    if not os.path.isdir(abs_studies_path):
        print(f"Error: Root directory '{studies_dir}' does not exist.")
        sys.exit(1)

    print("--- Optuna Study Manager (SQLite Version) ---")
    browse_studies(abs_studies_path)
    print("\nExiting manager. Goodbye!")


if __name__ == "__main__":
    main()