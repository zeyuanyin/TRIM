PROJECT_DIR=$(pwd)

# Pytorch
# pip3 install -i https://download.pytorch.org/whl/cu124 -U torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
# pip3 install -i https://download.pytorch.org/whl/cu124 xformers --no-deps

pip3 install -U torch torchvision torchaudio
pip3 install -U xformers

# A modified gaussian splatting (+ alpha, depth, normal rendering)
cd src && git clone https://github.com/BaowenZ/RaDe-GS.git --recursive && cd RaDe-GS/submodules
pip3 install ./diff-gaussian-rasterization
cd ${PROJECT_DIR}

# Others
pip3 install -U gpustat
pip3 install -U -r requirements.txt
pip3 install tqdm
# sudo apt-get install -y ffmpeg
