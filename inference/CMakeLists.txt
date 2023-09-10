add_executable(cat_dog "cat_dog.cpp")
target_include_directories(cat_dog PRIVATE /data/zyn_data/envs/onnxruntime/onnxruntime-linux-x64-1.15.1/include)
target_link_libraries(cat_dog PRIVATE onnxruntime png)
target_link_directories(cat_dog PRIVATE /data/zyn_data/envs/onnxruntime/onnxruntime-linux-x64-1.15.1/lib)
