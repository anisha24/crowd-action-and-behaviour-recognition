sudo yum update -y

curl -O https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
bash Anaconda3-2024.02-1-Linux-x86_64.sh
eval "$(/root/anaconda3/bin/conda shell.bash hook)"
conda env create -f environment.yml
rm -rf Anaconda3-2024.02-1-Linux-x86_64.sh

pip install opencv-python-headless
yum install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
yum install epel-release -y
yum install python3-opencv opencv-devel -y
yum install mesa-libGL -y
yum install libglvnd-opengl.x86_64 -y
yum install libglvnd-opengl.i686 -y