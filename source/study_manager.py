import os
import sys
import optuna
import pandas as pd
from typing import Optional

# noinspection PyProtectedMember
from optuna.storages._rdb.models import TrialModel



studies_dir = "./data/studies"

# Increase pandas display width for better readability of dataframes
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)



def load_study(study_path: str) -> Optional[optuna.Study]:
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
    print("\n--- Study Summary ---")
    print(f"Study Name: {study.study_name}")
    print(f"Direction: {study.direction.name}")
    print(f"Number of trials: {len(study.trials)}")

    try:
        print("\n--- Best Trial ---")
        best = study.best_trial
        print(f"\tNumber: {best.number}")
        print(f"\tValue: {best.value:.6f}")
        print("\tParams: ")
        for key, value in best.params.items():
            print(f"\t\t{key}: {value}")
    except ValueError:
        print("No completed trials found in this study yet.")

    print("\n--- All Trials ---")
    print(study.trials_dataframe().set_index('number'))
    print("-" * 20)



def delete_trial_from_db(study: optuna.Study) -> bool:
    """
    Asks for a trial number and permanently deletes it from the database
    by performing a direct SQLAlchemy delete operation.
    """
    view_study_summary(study)
    try:
        trial_number_str = input("Enter the number of the trial to DELETE (or 'c' to cancel): ").strip().lower()
        if trial_number_str == 'c':
            return False

        trial_number_to_delete = int(trial_number_str)
        trial_to_delete = next((t for t in study.trials if t.number == trial_number_to_delete), None)

        if trial_to_delete is None:
            print(f"Error: Trial number {trial_number_to_delete} not found.")
            return False

        confirm = input(
            f"WARNING: This will PERMANENTLY delete Trial #{trial_number_to_delete} from the database.\n"
            "This action cannot be undone. Are you sure? (y/n): "
        ).lower()

        if confirm == 'y':
            # noinspection PyProtectedMember
            trial_id_to_delete = trial_to_delete._trial_id
            # noinspection PyProtectedMember, PyUnresolvedReferences
            backend = study._storage._backend
            session = backend.scoped_session()
            trial_model_to_delete = (
                session.query(TrialModel)
                .filter(TrialModel.trial_id == trial_id_to_delete)
                .one()
            )
            session.delete(trial_model_to_delete)
            session.commit()
            print(f"\nTrial #{trial_number_to_delete} has been permanently deleted.")
            return True
        else:
            print("Deletion cancelled.")
            return False

    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"An error occurred during deletion: {e}")
    return False



def manage_study_menu(study_path: str):
    """Main interactive menu for managing a single selected study."""
    while True:
        study = load_study(study_path)
        if not study: input("Press Enter to return to file browser..."); return

        print("\n" + "=" * 50)
        print(f"Managing Study: {study.study_name}")
        print(f"File: {os.path.basename(study_path)}")
        print("=" * 50)
        print("v) View summary")
        print("d) Delete a trial (Permanent)")
        print("b) Back to file browser")
        print("q) Quit")

        choice = input("> ").strip().lower()

        if   choice == 'v': view_study_summary(study); input("\nPress Enter to continue...")
        elif choice == 'd': delete_trial_from_db(study)
        elif choice == 'b': break
        elif choice == 'q': print("Exiting tool."); sys.exit(0)
        else: print("Invalid choice.")



def browse_studies(current_path: str):
    while True:
        print("\n" + "=" * 60)
        print(f"Current Directory: {os.path.relpath(current_path, studies_dir)}")
        print("=" * 60)

        try: items = sorted(os.listdir(current_path))
        except FileNotFoundError: print(f"Error: Directory not found: {current_path}"); return

        dirs = [d for d in items if os.path.isdir(os.path.join(current_path, d))]
        db_files = [f for f in items if f.endswith(".db")]

        print("Please make a selection:\n")
        for i, dirname in enumerate(dirs): print(f"\t{i + 1:2d}) [DIR]\t{dirname}")

        if db_files: print("-" * 20)
        for i, filename in enumerate(db_files): print(f"\t{len(dirs) + i + 1:2d}) [STUDY] {filename}")

        print("\n" + "-" * 20)
        if current_path != os.path.abspath(studies_dir): print("\tb) Back")
        print("\tq) Quit")

        choice = input("\nEnter your choice: ").strip().lower()

        if choice == 'q': break
        elif choice == 'b' and current_path != os.path.abspath(studies_dir):
            current_path = os.path.dirname(current_path)
            continue

        try:
            choice_index = int(choice) - 1
            if not (0 <= choice_index < len(dirs) + len(db_files)): raise ValueError
            if choice_index < len(dirs):
                current_path = os.path.join(current_path, dirs[choice_index])
            else:
                file_to_manage = db_files[choice_index - len(dirs)]
                full_path = os.path.join(current_path, file_to_manage)
                manage_study_menu(full_path)
        except ValueError:
            print("\n*** Invalid choice. Please try again. ***")



def main():
    abs_studies_path = os.path.abspath(studies_dir)
    if not os.path.isdir(abs_studies_path):
        print(f"Error: Root directory '{studies_dir}' does not exist.")
        sys.exit(1)

    print("--- Optuna Study Manager ---")
    browse_studies(abs_studies_path)
    print("\nExiting manager. Goodbye!")



if __name__ == "__main__":
    main()