install-uv:
# Install uv python package manager
# see https://docs.astral.sh/uv/getting-started/installation/
	curl -LsSf https://astral.sh/uv/install.sh | sh

venv-create:
	uv venv

venv-install:
# can also use `uv pip install -r pyproject.toml`,
# however this will keep existing packages from current venv compatible with the lock file.
# see: https://docs.astral.sh/uv/pip/compile/#syncing-an-environment
# the better approach to ensure total sync between the lockfile and venv is:
	uv sync

server-install-system-deps:
	sudo add-apt-repository -y ppa:jonathonf/ffmpeg-4
	sudo apt update
	sudo apt install -y ffmpeg
	sudo apt-get install -y git-lfs
	sudo apt-get install -y tmux

# TODO: ensure it's needed
server-hf-hub-setup:
	huggingface-cli login
	git config --global credential.helper store
	@echo "NOTE: To be able to push model checkpoints to huggingface hub, please login into git first:"
	@echo "> git config --globase user.name <user_name>"
	@echo "> git config --globase user.email <user_email>"
