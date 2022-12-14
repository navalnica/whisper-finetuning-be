sudo add-apt-repository -y ppa:jonathonf/ffmpeg-4
sudo apt update
sudo apt install -y ffmpeg

sudo apt-get install git-lfs

sudo apt-get install tmux

cd ~
echo "executing env setup from $(pwd)"

python3 -m venv ~/python_venvs/hf_env
source ~/python_venvs/hf_env/bin/activate
echo "source ~/python_venvs/hf_env/bin/activate" >> ~/.bashrc

git clone https://github.com/yks72p/whisper-finetuning-be
pip install -r ~/whisper-finetuning-be/requirements.txt

git config --global credential.helper store
huggingface-cli login

echo "env setup"
echo "! PLEASE LOGIN INTO GIT TO BE ABLE TO PUSH TO HF HUB !"
echo "> git config --globase user.name <user_name>"
echo "> git config --globase user.email <user_email>"
