python ./MCU-MixQ/
mkdir ./MCU-MixQ/TinyEngine/Src/TinyEngine/include/arm_cmsis
cp -r ./MCU-MixQ/TinyEngine/third_party/CMSIS/CMSIS/NN/Include/*.h ./tutorial/TinyEngine_vww_tutorial/Src/TinyEngine/include/arm_cmsis
cp -r ./MCU-MixQ/TinyEngine/third_party/CMSIS/CMSIS/DSP/Include/dsp ./tutorial/TinyEngine_vww_tutorial/Src/TinyEngine/include/arm_cmsis
cp ./MCU-MixQ/TinyEngine/third_party/CMSIS/CMSIS/DSP/Include/arm_common_tables.h ./tutorial/TinyEngine_vww_tutorial/Src/TinyEngine/include/arm_cmsis
cp ./MCU-MixQ/TinyEngine/third_party/CMSIS/CMSIS/DSP/Include/arm_math_memory.h ./tutorial/TinyEngine_vww_tutorial/Src/TinyEngine/include/arm_cmsis
cp ./MCU-MixQ/TinyEngine/third_party/CMSIS/CMSIS/DSP/Include/arm_math_types.h ./tutorial/TinyEngine_vww_tutorial/Src/TinyEngine/include/arm_cmsis
cp ./MCU-MixQ/TinyEngine/third_party/CMSIS/CMSIS/DSP/Include/arm_math.h ./tutorial/TinyEngine_vww_tutorial/Src/TinyEngine/include/arm_cmsis
cp -r ./MCU-MixQ/TinyEngine/third_party/CMSIS/CMSIS/Core/Include ./tutorial/TinyEngine_vww_tutorial/Drivers/CMSIS