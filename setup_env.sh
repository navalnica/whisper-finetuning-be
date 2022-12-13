sudo add-apt-repository -y ppa:jonathonf/ffmpeg-4
sudo apt update
sudo apt install -y ffmpeg

sudo apt-get install git-lfs

python3 -m venv hf_env
source hf_env/bin/activate
echo "source ~/hf_env/bin/activate" >> ~/.bashrc

git clone https://github.com/huggingface/community-events.git
pip install -r community-events/whisper-fine-tuning-event/requirements.txt

git config --global credential.helper store
huggingface-cli login
