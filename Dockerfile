FROM openai/retro-agent:tensorflow

# Needed for OpenCV.
RUN apt-get update && \
    apt-get install -y libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/*

# Baselines has some unneeded and cumbersome dependencies,
# so we manually fetch the deps we need.
RUN . ~/venv/bin/activate && \
    pip install scipy tqdm joblib zmq dill progressbar2 cloudpickle opencv-python && \
    pip install --no-deps git+https://github.com/openai/baselines.git

# Use the anyrl open source RL framework.
RUN . ~/venv/bin/activate && \
    pip install anyrl==0.11.17

ADD RainbowAgent.py ./agent.py
ADD custom_sonic_util.py .

CMD ["python", "-u", "/root/compo/agent.py"]
