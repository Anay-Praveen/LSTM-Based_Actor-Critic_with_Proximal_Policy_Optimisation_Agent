== Installation ==

Install the required Python libraries using pip. It's recommended to use a virtual environment.

pip install nasim
pip install pandas
pip install numpy
pip install matplotlib
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

(Ensure you have a compatible CUDA setup if using the GPU version of PyTorch specified above. Adjust the PyTorch command if using CPU or a different CUDA version.)

== Training ==
== Performance Note ==
Training the full curriculum takes approximately 3.5 to 4 hours on an NVIDIA RTX 4070 Super GPU.
On hardware with lower specifications, expect significantly longer training times, potentially 5 hours or more.

To train the agent through the curriculum:

Run: python main.py

== Testing Pre-trained Model ==

To test a previously trained model:
Run: python test_agent.py

