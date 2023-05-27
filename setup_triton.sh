apt update -y
apt upgrade -y
apt install vim -y

wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-11.8_1.0-1_amd64.deb
os="ubuntu2004"
tag="8.6.1-cuda-11.8"
dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get install tensorrt -y

# install cuda-tool-kit
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
#mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
#wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
#dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
#cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
#apt-get update
#apt-get -y install cuda
export PATH="/usr/local/cuda-11/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11/lib64:$LD_LIBRARY_PATH"
source ~/.bashrc

curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | tee /etc/apt/sources.list.d/nvhpc.list
apt-get update -y
apt-get install -y nvhpc-23-5-cuda-multi
wget https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.1/local_installers/11.8/cudnn-linux-x86_64-8.9.1.23_cuda11-archive.tar.xz/
tar -xvf cudnn-linux-x86_64-8.9.1.23_cuda11-archive.tar.xz
cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*


git clone https://github.com/triton-inference-server/server.git
cd sever
./build.py -v --no-container-build --build-dir=`pwd`/build --enable-all