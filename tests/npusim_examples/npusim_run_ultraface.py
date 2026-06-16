# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_v2_sim_utils import CoralNPUV2Simulator
import numpy as np
import os

def run_ultraface():
    print(f"Running UltraFace...")
    npu_sim = CoralNPUV2Simulator(highmem_ld=True, exit_on_ebreak=True)
    r = runfiles.Create()

    # Locate the compiled UltraFace .elf in the runfiles tree
    elf_file = r.Rlocation('coralnpu_hw/tests/npusim_examples/run_ultraface_binary.elf')

    symbols_to_fetch = [
        'inference_status',
        'inference_input',
        'inference_output_scores',
        'inference_output_boxes',
        'model_buffer'
    ]

    entry_point, symbol_map = npu_sim.get_elf_entry_and_symbol(elf_file, symbols_to_fetch)
    npu_sim.load_program(elf_file, entry_point)

    # --- FLASH THE MODEL INTO MEMORY ---
    if symbol_map.get('model_buffer'):
        # Use runfiles to locate the file dynamically inside Bazel's sandbox
        model_path = r.Rlocation('coralnpu_hw/tests/cocotb/tutorial/tfmicro/models/version-slim-320_static_int8.tflite')
        
        # Fallback check to provide a clean error message if Bazel doesn't map it
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError("Bazel could not find the .tflite file in the runfiles tree. Check your BUILD file 'data' section.")

        print(f"Loading model from {model_path} into memory...")
        with open(model_path, 'rb') as f:
            model_data = f.read()
            
        # FIXED: Convert raw bytes to a numpy uint8 array
        model_array = np.frombuffer(model_data, dtype=np.uint8)
        npu_sim.write_memory(symbol_map['model_buffer'], model_array)
    # ----------------------------------------

    if symbol_map.get('inference_input'):
        # Input size: 240 (H) x 320 (W) x 3 (Channels)
        input_data = np.random.randint(-128, 127, size=(240 * 320 * 3,), dtype=np.int8)
        npu_sim.write_memory(symbol_map['inference_input'], input_data)

    print("Running simulation...", flush=True)
    npu_sim.run()
    npu_sim.wait()
    print(f"Cycles taken by the simulation: {npu_sim.get_cycle_count()}")

    if symbol_map.get('inference_output_scores'):
        # Read the scores memory map (4420 anchors * 2 classes)
        output_scores = npu_sim.read_memory(symbol_map['inference_output_scores'], 4420 * 2)
        output_scores = np.array(output_scores, dtype=np.int8)
        print(f"Scores array preview: {output_scores[:10]}")

    if symbol_map.get('inference_output_boxes'):
        # Read the bounding boxes memory map (4420 anchors * 4 coordinates)
        output_boxes = npu_sim.read_memory(symbol_map['inference_output_boxes'], 4420 * 4)
        output_boxes = np.array(output_boxes, dtype=np.int8)
        print(f"Boxes array preview: {output_boxes[:12]}")

    if symbol_map.get('inference_status'):
        inference_status = npu_sim.read_memory(symbol_map['inference_status'], 1)[0]
        print(f"inference_status: {inference_status}")

if __name__ == "__main__":
    run_ultraface()
