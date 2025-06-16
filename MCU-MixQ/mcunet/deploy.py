import argparse
from code_generator.CodegenUtilTFlite import GenerateSourceFilesFromTFlite
from mcunet.mcunet.model_zoo import download_tflite

parser = argparse.ArgumentParser(description="Deploy a model to MCU")
parser.add_argument(
    "--model",
    type=str,
)

args = parser.parse_args()
model_path = args.model
# 3. Let's generate source code for on-device deployment
peakmem = GenerateSourceFilesFromTFlite(
    model_path,
    life_cycle_path="./lifecycle.png",
)