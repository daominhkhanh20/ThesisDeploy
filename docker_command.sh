bash -c "tail -f /dev/null"

bash -c "/usr/sbin/sshd -D"
nvcr.io/nvidia/tritonserver:21.06.1-py3

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html



bash -c "apt update;apt install -y wget;DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y;mkdir -p ~/.ssh;cd $_;chmod 700 ~/.ssh;echo ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKvhxW8+MW8HydnawgFvHCMJ575EyQBdMlIQ1IrmBZaZ khanhc1k36@gmail.com > authorized_keys;chmod 700 authorized_keys;service ssh start;sleep infinity"