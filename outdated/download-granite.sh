curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | bash && \
    dnf install -y git-lfs

git clone https://huggingface.co/instructlab/granite-7b-lab
