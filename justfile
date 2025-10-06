# Using variables for executables from the virtual environment
python := "./.venv/bin/python"
pip := "./.venv/bin/pip"
rscript := "Rscript"

# Dynamically find the site-packages directory to avoid hardcoding python versions.
# More robust across different environments.
site_packages := `./.venv/bin/python -c "import sysconfig; print(sysconfig.get_path('purelib'))"`

patches_dir := "./source/patches/"


# --- Main Recipes ---

dependencies:
    bash ./dependencies.sh


# pip

pip_install PACKAGE:
    {{ pip }} install {{ PACKAGE }}

pip_upgrade PACKAGE:
    {{ pip }} install --upgrade {{ PACKAGE }}

pip_freeze:
    {{ pip }} freeze > requirements.txt

pip_install_all:
    {{ pip }}  install -r requirements.txt

pip_uninstall_all:
    {{ pip }} uninstall -r requirements.txt -y

pip_reinstall_all: pip_uninstall_all pip_install_all

pip_deep_clean:
    {{ pip }} cache purge
    @rm -rf .venv


# Python scripts for project (mostly used for local debugging)

encodings:  # 1
    {{ python }} ./source/data_scripts/encodings.py

train_model: # 2
    {{ python }} ./source/training/train_model.py

evaluation: # 3
    {{ python }} ./source/evaluation/evaluation.py

hyper_param:
    {{ python }} ./source/training/hyper_param.py


# Tools

study_manager:
    {{ python }} ./source/study_manager.py

display_figures:
    {{ python }} ./source/display_figures.py


# R scripts

model_performances:
    {{ rscript }} ./r_scripts/model_performances.R



# GENERIC PATCH HELPER RECIPES

# Saves the original version of a file from site-packages.
# Usage: _save_original <source_file> <destination_copy>
_save_original source_file original_copy:
    @echo "Saving original {{ source_file }} to {{ original_copy }}..."
    @mkdir -p {{ patches_dir }}
    @cp "{{ source_file }}" "{{ original_copy }}"
    @echo "Original file saved."

# Creates a .patch file by comparing the original and the modified versions.
# Usage: _create_patch <original_copy> <modified_file> <patch_output_file>
_create_patch original_copy modified_file patch_file:
    @echo "Creating patch file: {{ patch_file }}"
    @diff -u "{{ original_copy }}" "{{ modified_file }}" > "{{ patch_file }}"
    @echo "Patch created successfully."

# Applies a patch to a target file.
# Usage: _apply_patch <target_file> <patch_file>
_apply_patch target_file patch_file:
    @echo "Applying patch {{ patch_file }} to {{ target_file }}..."
    @patch --forward "{{ target_file }}" < "{{ patch_file }}"
    @echo "Patch applied successfully."

# Reverts a patch from a target file.
# Usage: _revert_patch <target_file> <patch_file>
_revert_patch target_file patch_file:
    @echo "Reverting patch on {{ target_file }} using {{ patch_file }}..."
    @patch -R "{{ target_file }}" < "{{ patch_file }}"
    @echo "Patch reverted successfully."


# --- Patch Wrapper ---

## <PACKAGE_NAME> <PATCH>.py Patch Management
#
## --- Path Variables ---
#<PACKAGE_NAME>_<FILE>_path := site_packages + "/<PACKAGE_DIR_NAME>/<FILE>.py"
#original_<FILE>_path := patches_dir + "<FILE>.py.original"
#modified_<FILE>_path := patches_dir + "<FILE>.py.modified"
#patch_<FILE>_path := patches_dir + "<FILE>.patch"
#
## --- Recipes ---
#<FILE>_save_original:
#    ./just _save_original '{{<PACKAGE_NAME>_<FILE>_path}}' '{{original_<FILE>_path}}'
#
#<FILE>_create_patch:
#    ./just _create_patch '{{original_<FILE>_path}}' '{{modified_<FILE>_path}}' '{{patch_<FILE>_path}}'
#
#<FILE>_patch:
#    ./just _apply_patch '{{<PACKAGE_NAME>_<FILE>_path}}' '{{patch_<FILE>_path}}'
#
#<FILE>_revert_patch:
#    ./just _revert_patch '{{<PACKAGE_NAME>_<FILE>_path}}' '{{patch_<FILE>_path}}'