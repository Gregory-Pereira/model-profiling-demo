# FROM quay.io/grpereir/granite-7b-lab:latest
FROM registry.access.redhat.com/ubi9/python-311:1-77.1729767833

USER root

ADD profile.py /opt/app-root/src
ADD requirements.txt /opt/app-root/src

RUN pip install -r /opt/app-root/src/requirements.txt

ENTRYPOINT ["python", "profile.py"]
